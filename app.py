from flask import Flask, request, jsonify, render_template, send_file, Response
import google.generativeai as genai
import os
import json
import pandas as pd
from dotenv import load_dotenv
import io  # For in-memory file handling
from werkzeug.utils import secure_filename # For secure filenames
import datetime # For timestamping results
import threading
import queue
import uuid
import time
import google.api_core.exceptions # Import exception for retries
import requests # For calling DeepSeek API

# Import evaluation functions
from evaluate import evaluate_results_background

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load DeepSeek API Key
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

app = Flask(__name__)

RESULTS_DIR = "analysis_results"
EVALUATION_DIR = "evaluation_outputs"

# Dictionary to keep track of active analysis tasks
# Structure: { task_id: { 'thread': threading.Thread, 'stop_event': threading.Event, 'queue': queue.Queue, 'status': str, 'results_df': pd.DataFrame, 'filename': str } }
active_tasks = {}

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(EVALUATION_DIR):
     os.makedirs(EVALUATION_DIR)

def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

# --- LLM Interaction Abstraction --- #

def call_llm(prompt_data, api_choice, task_type):
    """Calls the selected LLM API (Gemini or DeepSeek) and returns a consistent response.
    
    Args:
        prompt_data (dict): Contains data needed for the specific API call 
                              (e.g., {'system_prompt': str, 'user_input': str} for stance).
        api_choice (str): 'gemini' or 'deepseek'.
        task_type (str): 'stance' or 'evaluation' (determines expected response format).

    Returns:
        dict: {'success': bool, 'content': dict or None, 'error': str or None}
              'content' will contain the parsed response (e.g., {'target': str, 'stance': str} for stance).
    """
    
    max_retries = 3
    base_delay = 1 # seconds
    last_exception_str = None
    
    for attempt in range(max_retries):
        try:
            if api_choice == 'gemini':
                # --- Gemini Call Logic --- # (Adapted from analyze_stance)
                if task_type == 'stance':
                    if not genai.config.is_configured(): # Ensure configured in this context
                         genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                         
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    chat = model.start_chat(history=[
                        {"role": "user", "parts": [prompt_data['system_prompt']]},
                        {"role": "model", "parts": ["I'll analyze text and respond with only a JSON object containing the target and stance."]}
                    ])
                    response = chat.send_message(prompt_data['user_input'])
                    response_text = response.text.strip()
                    
                    # Extract JSON
                    if not response_text.startswith('{'):
                        import re
                        json_match = re.search(r'({.*})', response_text, re.DOTALL)
                        if json_match:
                            response_text = json_match.group(1)
                    
                    parsed_response = json.loads(response_text)
                    
                    # Basic validation (more detailed in analyze_stance)
                    if 'target' in parsed_response and 'stance' in parsed_response:
                        return {'success': True, 'content': parsed_response, 'error': None}
                    else:
                        raise ValueError("Gemini response missing target or stance.")
                else:
                    # Handle other task types for Gemini later (e.g., evaluation)
                    return {'success': False, 'content': None, 'error': f"Gemini task type '{task_type}' not implemented in call_llm."}

            elif api_choice == 'deepseek':
                # --- DeepSeek Call Logic --- #
                if task_type == 'stance':
                    if not deepseek_api_key:
                        return {'success': False, 'content': None, 'error': "DeepSeek API key not configured."}
                        
                    headers = {
                        'Authorization': f'Bearer {deepseek_api_key}',
                        'Content-Type': 'application/json'
                    }
                    # DeepSeek uses messages format similar to OpenAI
                    payload = {
                        "model": "deepseek-chat", # Or deepseek-coder if preferred
                        "messages": [
                            {"role": "system", "content": prompt_data['system_prompt']},
                            {"role": "user", "content": prompt_data['user_input']}
                        ],
                         "stream": False, # We want the full response
                         "response_format": { "type": "json_object" } # Request JSON output
                    }
                    
                    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60) # Add timeout
                    response.raise_for_status() # Raise HTTPError for bad status codes (4xx or 5xx)
                    
                    response_data = response.json()
                    
                    # Check for DeepSeek-specific errors if any in response body
                    if 'error' in response_data:
                        raise Exception(f"DeepSeek API Error: {response_data['error']}")
                        
                    # Extract content (assuming it follows OpenAI structure)
                    message_content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if not message_content:
                         raise ValueError("DeepSeek response did not contain message content.")
                         
                    # Parse the JSON string within the content
                    parsed_response = json.loads(message_content)
                    
                    # Basic validation
                    if 'target' in parsed_response and 'stance' in parsed_response:
                        return {'success': True, 'content': parsed_response, 'error': None}
                    else:
                        raise ValueError("DeepSeek response missing target or stance.")
                else:
                     # Handle other task types for DeepSeek later
                     return {'success': False, 'content': None, 'error': f"DeepSeek task type '{task_type}' not implemented in call_llm."}
            
            else:
                return {'success': False, 'content': None, 'error': f"Unknown api_choice: {api_choice}"}

        # --- Exception Handling with Retries --- #
        except (google.api_core.exceptions.DeadlineExceeded, google.api_core.exceptions.ResourceExhausted, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            last_exception_str = str(e)
            status_code = getattr(e, 'response', None) and getattr(e.response, 'status_code', None)
            error_type = type(e).__name__
            
            # Specific check for DeepSeek 429/500/503 for retry
            should_retry = True
            if isinstance(e, requests.exceptions.HTTPError):
                 if status_code in [429, 500, 503]: # Rate limit or server error
                     logger_msg = f"LLM Call Attempt {attempt + 1} failed: HTTP {status_code} ({error_type}). Retrying..."
                 else:
                     logger_msg = f"LLM Call Attempt {attempt + 1} failed: HTTP {status_code} ({error_type}). Not retrying."
                     should_retry = False
            elif isinstance(e, google.api_core.exceptions.ResourceExhausted):
                 logger_msg = f"LLM Call Attempt {attempt + 1} failed: Gemini Rate Limit/Quota. Retrying..."
            elif isinstance(e, google.api_core.exceptions.DeadlineExceeded) or isinstance(e, requests.exceptions.Timeout):
                 logger_msg = f"LLM Call Attempt {attempt + 1} failed: Timeout ({error_type}). Retrying..."
            else: # Other RequestException
                 logger_msg = f"LLM Call Attempt {attempt + 1} failed: Network/Request Error ({error_type}). Retrying..."
            
            app.logger.warning(logger_msg)
            
            if should_retry and attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                 app.logger.error(f"LLM Call failed after {max_retries} attempts or non-retryable error. Last error: {last_exception_str}")
                 return {'success': False, 'content': None, 'error': f"API Error after retries: {last_exception_str}"}

        except (json.JSONDecodeError, ValueError, KeyError) as e:
             # Catch parsing errors or missing keys
             last_exception_str = f"Response parsing error: {str(e)}"
             app.logger.error(f"LLM Call failed: {last_exception_str}")
             return {'success': False, 'content': None, 'error': last_exception_str}
             
        except Exception as e:
            # Catch any other unexpected errors
            last_exception_str = f"Unexpected error: {str(e)}"
            app.logger.error(f"LLM Call failed: {last_exception_str}", exc_info=True)
            return {'success': False, 'content': None, 'error': last_exception_str}

    # Should not be reached if logic is correct
    return {'success': False, 'content': None, 'error': f"LLM Call failed after {max_retries} attempts (unknown reason). Last error: {last_exception_str}"}

def analyze_stance(input_text, api_choice='gemini'):
    if not input_text or not isinstance(input_text, str) or input_text.strip() == "":
        return {"target": "ERROR", "stance": "Input text invalid or empty"}
        
    # --- Define prompts --- #
    system_prompt = """
    You are a stance detection system. Analyze the given text and identify:
    1. The target - what the text is discussing or taking a stance on. IMPORTANT: The target must be only 1-4 words maximum.
    2. The stance - whether the author is in FAVOR, AGAINST, or NEUTRAL toward the target
    
    You MUST respond ONLY with a valid JSON object in the following format:
    {
      "target": "identified target (1-4 words only)",
      "stance": "STANCE"
    }
    
    Where STANCE is one of: FAVOR, AGAINST, or NEUTRAL.
    Do not include any explanation or additional text in your response.
    """
    prompt_data = {
        'system_prompt': system_prompt,
        'user_input': input_text
    }

    # --- Call the LLM abstraction --- #
    # In a full implementation, api_choice would come from the request
    llm_response = call_llm(prompt_data, api_choice, 'stance') 

    # --- Process the standardized response --- #
    if not llm_response['success']:
        # Return error from call_llm
        return {"target": "LLM_ERROR", "stance": llm_response['error']}

    # --- Post-process and validate the successful content --- #    
    parsed_response = llm_response['content']
    try:
        # Ensure required fields are present (redundant but safe)
        if 'target' not in parsed_response or 'stance' not in parsed_response:
            return {"target": "LLM_ERROR", "stance": "API response missing target/stance after call."} 
            
        # Validate target length (1-4 words)
        target_words = parsed_response['target'].split()
        if len(target_words) > 4:
            parsed_response['target'] = ' '.join(target_words[:4])
            
        # Normalize stance to uppercase
        parsed_response['stance'] = parsed_response['stance'].upper()
        
        # Validate stance value
        valid_stances = ["FAVOR", "AGAINST", "NEUTRAL"]
        if parsed_response['stance'] not in valid_stances:
             app.logger.warning(f"Invalid stance '{parsed_response['stance']}' received, defaulting to NEUTRAL.")
             parsed_response['stance'] = "NEUTRAL"
        
        return parsed_response # Return the processed content
        
    except Exception as e:
        # Catch unexpected errors during post-processing
        app.logger.error(f"Error post-processing LLM response: {str(e)}")
        return {"target": "PROCESS_ERROR", "stance": f"Error processing successful LLM response: {str(e)}"}

def process_csv_background(task_id, file_stream_data, filename, text_column, gt_target_column, gt_stance_column, start_row, api_choice):
    """Background task to process the CSV and send progress."""
    task = active_tasks.get(task_id)
    if not task:
        print(f"Task {task_id} not found for background processing.")
        return

    q = task['queue']
    stop_event = task['stop_event']
    results_list = []
    
    try:
        # Need to re-open the stream or pass data correctly
        # For simplicity, let's assume file_stream_data is bytes
        file_stream = io.BytesIO(file_stream_data) 
        df = pd.read_csv(file_stream)
        task['status'] = 'processing'
        total_rows = len(df)
        q.put(json.dumps({"type": "start", "total_rows": total_rows}))

        # Check if the specified text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
            
        # Adjust start_row for 0-based indexing
        start_index = start_row - 1 

        request_count = 0 # Counter for API requests
        pause_interval = 15 # Pause after every N requests
        pause_duration = 60 # Pause duration in seconds (1 minute)

        for index, row in df.iterrows():
            # Skip rows before the starting index
            if index < start_index:
                continue

            if stop_event.is_set():
                task['status'] = 'stopped'
                q.put(json.dumps({"type": "status", "message": f"Processing stopped by user at row {index + 1}."}))
                break

            input_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            # Put progress before calling LLM
            q.put(json.dumps({
                "type": "progress", 
                "row": index + 1, 
                "total": total_rows,
                "text": (input_text[:75] + '...') if len(input_text) > 75 else input_text
            }))
            
            analysis_result = analyze_stance(input_text, api_choice=api_choice)
            request_count += 1 # Increment after successful API call
            
            # Put result after calling LLM
            q.put(json.dumps({
                "type": "result", 
                "row": index + 1, 
                "target": analysis_result.get('target'),
                "stance": analysis_result.get('stance') 
            }))

            gt_target = str(row[gt_target_column]) if gt_target_column in row and pd.notna(row[gt_target_column]) else "N/A"
            gt_stance = str(row[gt_stance_column]) if gt_stance_column in row and pd.notna(row[gt_stance_column]) else "N/A"

            results_list.append({
                'Original Text': input_text,
                'Predicted Target': analysis_result.get('target', 'ERROR'),
                'Predicted Stance': analysis_result.get('stance', 'ERROR'),
                'Ground Truth Target': gt_target,
                'Ground Truth Stance': gt_stance,
                'Raw Model Response': analysis_result.get('raw_response', '')
            })
            
            # --- Rate Limiting Pause --- #
            if request_count % pause_interval == 0 and index + 1 < total_rows: # Check if pause needed and not the last row
                try:
                    pause_msg = f"Pausing for {pause_duration} seconds to avoid rate limits after {request_count} requests..."
                    q.put(json.dumps({"type": "status", "message": pause_msg}))
                    
                    # Sleep while allowing stop signal check
                    for _ in range(pause_duration):
                        if stop_event.is_set():
                            q.put(json.dumps({"type": "status", "message": "Pause interrupted by stop signal."}))
                            break # Break inner sleep loop
                        time.sleep(1)
                    else: # Only runs if inner loop completes without break
                         q.put(json.dumps({"type": "status", "message": "Resuming analysis..."}))
                         continue # Continue outer loop
                    
                    # If break happened in inner loop, break outer loop too
                    break
                except Exception as sleep_err:
                     q.put(json.dumps({"type": "error", "message": f"Error during pause: {str(sleep_err)}"}))
            # --- End Rate Limiting Pause --- #

        else: # Only runs if loop completes without break
            task['status'] = 'completed'
            q.put(json.dumps({"type": "status", "message": "Processing completed successfully."}))

    except Exception as e:
        task['status'] = 'error'
        error_message = f"Error during processing: {str(e)}"
        q.put(json.dumps({"type": "error", "message": error_message}))
        app.logger.error(f"Task {task_id} error: {error_message}")

    finally:
        # Save whatever results were collected
        if results_list:
            results_df = pd.DataFrame(results_list)
            task['results_df'] = results_df # Store for potential later use
            
            # Save final/partial results CSV
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            status_suffix = "stopped" if task['status'] == 'stopped' else "completed"
            save_base_filename = secure_filename(filename.rsplit('.', 1)[0])
            save_filename = f"{save_base_filename}_{timestamp}_{status_suffix}.csv"
            save_path = os.path.join(RESULTS_DIR, save_filename)
            try:
                results_df.to_csv(save_path, index=False)
                q.put(json.dumps({"type": "saved", "filename": save_filename}))
            except Exception as e:
                 save_error = f"Failed to save results file {save_filename}: {str(e)}"
                 q.put(json.dumps({"type": "error", "message": save_error}))
                 app.logger.error(f"Task {task_id} save error: {save_error}")
        else:
             q.put(json.dumps({"type": "status", "message": "No results generated."}))
        
        # Signal completion/stop to the SSE stream
        q.put("__DONE__" if task['status'] != 'error' else "__ERROR__")
        
        # Clean up task entry after a delay? Or maybe leave it for inspection?
        # For now, leave it.
        print(f"Background task {task_id} finished with status: {task['status']}")

@app.route('/stream_progress/<task_id>')
def stream_progress(task_id):
    task = active_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
        
    q = task['queue']

    def event_stream():
        while True:
            message = q.get() # Blocks until a message is available
            if message in ["__DONE__", "__ERROR__", None]: # Termination signal
                yield f"event: close\ndata: {message}\n\n"
                # Optionally remove task from active_tasks here or after a delay
                # if task_id in active_tasks:
                #     del active_tasks[task_id]
                # print(f"SSE stream closed for task {task_id}")
                break
            try:
                # Ensure message is JSON before sending
                json.loads(message) # Validate it's JSON
                yield f"data: {message}\n\n"
            except json.JSONDecodeError:
                # If it's not JSON (maybe unexpected internal message?), log and skip
                app.logger.warning(f"Non-JSON message in queue for task {task_id}: {message}")
            except Exception as e:
                app.logger.error(f"Error yielding SSE message for task {task_id}: {e}")
                yield f"event: error\ndata: Error processing message stream\n\n"
                break # Stop streaming on error

    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/stop_analysis/<task_id>', methods=['POST'])
def stop_analysis(task_id):
    task = active_tasks.get(task_id)
    if task:
        task['stop_event'].set() # Signal the thread to stop
        task['status'] = 'stopping'
        return jsonify({"message": "Stop signal sent to task."}) 
    else:
        return jsonify({"error": "Task not found or already completed."}), 404

@app.route('/detect_stance', methods=['POST'])
def detect_stance():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    input_text = data['text']
    api_choice = data.get('api_choice', 'gemini') # Get API choice from JSON
    result = analyze_stance(input_text, api_choice=api_choice)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

# --- API Endpoints for Managing Results --- #

@app.route('/list_results', methods=['GET'])
def list_results():
    try:
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
        files.sort(reverse=True) # Show newest first
        return jsonify({"results": files})
    except Exception as e:
        return jsonify({"error": f"Could not list results: {str(e)}"}), 500

@app.route('/view_result/<filename>', methods=['GET'])
def view_result(filename):
    safe_filename = secure_filename(filename) # Sanitize filename
    file_path = os.path.join(RESULTS_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        df = pd.read_csv(file_path)
        
        # Convert DataFrame directly to a JSON string, handling NaN -> null
        # pandas' to_json handles NaN correctly by default
        json_string = df.to_json(orient='records', date_format='iso')
        
        # Load the valid JSON string back into a Python object 
        # This ensures the structure is correct before jsonify
        results_data = json.loads(json_string)
        
        return jsonify({"data": results_data}) # Now jsonify receives a standard Python object
    except Exception as e:
        return jsonify({"error": f"Could not read or process file: {str(e)}"}), 500

@app.route('/delete_result/<filename>', methods=['DELETE'])
def delete_result(filename):
    safe_filename = secure_filename(filename)
    file_path = os.path.join(RESULTS_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        os.remove(file_path)
        return jsonify({"message": f"File '{safe_filename}' deleted successfully."})
    except Exception as e:
        return jsonify({"error": f"Could not delete file: {str(e)}"}), 500

# --- Evaluation Results Management Endpoints --- #

@app.route('/list_evaluations', methods=['GET'])
def list_evaluations():
    """Lists evaluation result CSV files."""
    try:
        files = [f for f in os.listdir(EVALUATION_DIR) if f.endswith('.csv')]
        files.sort(reverse=True) # Show newest first
        return jsonify({"results": files})
    except Exception as e:
        return jsonify({"error": f"Could not list evaluation results: {str(e)}"}), 500

@app.route('/view_evaluation/<filename>', methods=['GET'])
def view_evaluation(filename):
    """Reads and returns the content of an evaluation result CSV, including averages."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(EVALUATION_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Evaluation file not found"}), 404
        
    try:
        df = pd.read_csv(file_path)
        json_string = df.to_json(orient='records', date_format='iso')
        results_data = json.loads(json_string)
        # Compute averages for Stance Match Pct and Target Match Pct
        stance_col = 'Stance Match Pct'
        target_col = 'Target Match Pct'
        stance_vals = pd.to_numeric(df[stance_col], errors='coerce') if stance_col in df else pd.Series(dtype=float)
        target_vals = pd.to_numeric(df[target_col], errors='coerce') if target_col in df else pd.Series(dtype=float)
        stance_avg = stance_vals[stance_vals >= 0].mean() if not stance_vals.empty else None
        target_avg = target_vals[target_vals >= 0].mean() if not target_vals.empty else None
        return jsonify({
            "data": results_data,
            "average": {
                "stance_match_pct": round(stance_avg, 2) if stance_avg is not None and not pd.isna(stance_avg) else None,
                "target_match_pct": round(target_avg, 2) if target_avg is not None and not pd.isna(target_avg) else None
            }
        })
    except Exception as e:
        return jsonify({"error": f"Could not read or process evaluation file: {str(e)}"}), 500

@app.route('/delete_evaluation/<filename>', methods=['DELETE'])
def delete_evaluation(filename):
    """Deletes an evaluation result CSV file."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(EVALUATION_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Evaluation file not found"}), 404
        
    try:
        os.remove(file_path)
        return jsonify({"message": f"Evaluation file '{safe_filename}' deleted successfully."})
    except Exception as e:
        return jsonify({"error": f"Could not delete evaluation file: {str(e)}"}), 500

# --- Evaluation Helper --- #
# --- Definition Moved to evaluate.py --- #

# --- Background Evaluation Worker --- #
# --- Definition Moved to evaluate.py --- # 

# --- Add back the Analysis Endpoint --- #
@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    text_column = request.form.get('text_column')
    gt_target_column = request.form.get('gt_target_column') or "GT Target" 
    gt_stance_column = request.form.get('gt_stance_column') or "GT Stance" 
    start_row = int(request.form.get('start_row', 1)) 
    api_choice = request.form.get('api_choice', 'gemini') # Get API choice

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV"}), 400
    if not text_column:
         return jsonify({"error": "Text column name must be provided"}), 400
    if start_row < 1:
        return jsonify({"error": "Start row must be 1 or greater"}), 400

    try:
        file_stream_data = file.read()
    except Exception as e:
         return jsonify({"error": f"Error reading uploaded file: {str(e)}"}), 500

    task_id = str(uuid.uuid4())
    task_queue = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=process_csv_background, 
        args=(task_id, file_stream_data, file.filename, text_column, gt_target_column, gt_stance_column, start_row, api_choice)
    )

    active_tasks[task_id] = {
        'type': 'analysis',
        'thread': thread,
        'stop_event': stop_event,
        'queue': task_queue,
        'status': 'starting',
        'filename': file.filename,
        'results_df': None 
    }

    thread.start()
    return jsonify({"message": "Analysis started", "task_id": task_id})
# --- End Analysis Endpoint --- #

# --- Evaluation Endpoints --- #

@app.route('/evaluate/<filename>')
def evaluate_page(filename):
    safe_filename = secure_filename(filename)
    # Basic check if file likely exists before rendering
    file_path = os.path.join(RESULTS_DIR, safe_filename)
    if not os.path.exists(file_path):
         # Render an error message or redirect
         # For simplicity, return an error string and status code
         return f"Error: Analysis results file not found: {safe_filename}", 404
    # If file exists, render the evaluation page template
    return render_template('evaluate.html', filename=safe_filename)

@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    data = request.json
    filename = data.get('filename')
    api_choice = data.get('api_choice', 'gemini') # Get API choice
    
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400
        
    safe_filename = secure_filename(filename)
    analysis_file_path = os.path.join(RESULTS_DIR, safe_filename)
    
    if not os.path.exists(analysis_file_path):
        return jsonify({"error": f"Analysis results file not found: {safe_filename}"}), 404

    task_id = str(uuid.uuid4())
    task_queue = queue.Queue()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=evaluate_results_background, # Call the imported function
        args=(
            task_id, 
            safe_filename,
            RESULTS_DIR,       # Pass results dir path
            EVALUATION_DIR,    # Pass evaluation dir path
            task_queue,        # Pass the queue
            stop_event,        # Pass the stop event
            app.logger,        # Pass the logger instance
            api_choice         # Pass API choice
            )
    )

    active_tasks[task_id] = {
        'type': 'evaluation', # Mark type
        'thread': thread,
        'stop_event': stop_event,
        'queue': task_queue,
        'status': 'starting',
        'filename': safe_filename, 
        'results_df': None 
    }

    thread.start()
    # Return the task_id so the frontend can connect to the stream
    return jsonify({"message": "Evaluation started", "task_id": task_id})

if __name__ == '__main__':
    app.run(debug=True) 
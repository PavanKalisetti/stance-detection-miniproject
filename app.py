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

# Import evaluation functions
from evaluate import evaluate_results_background

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

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

def analyze_stance(input_text):
    if not input_text or not isinstance(input_text, str) or input_text.strip() == "":
        return {"target": "ERROR", "stance": "Input text invalid or empty"}
    # System prompt for stance detection with structured output
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
    
    # Generate response from Gemini
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [system_prompt]
        },
        {
            "role": "model",
            "parts": ["I'll analyze text and respond with only a JSON object containing the target and stance."]
        }
    ])
    
    # --- Add Retry Logic --- #
    max_retries = 3
    base_delay = 1 # seconds
    response = None
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = chat.send_message(input_text)
            # If successful, break the loop
            break 
        except google.api_core.exceptions.DeadlineExceeded as e:
            last_exception = e
            app.logger.warning(f"Attempt {attempt + 1} failed: DeadlineExceeded. Retrying in {base_delay * (2**attempt)}s...")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                 app.logger.error(f"All {max_retries} attempts failed for DeadlineExceeded.")
                 # Let the exception propagate or return specific error
                 return {"target": "ERROR", "stance": f"API Timeout after {max_retries} retries ({str(e)})"}
        except Exception as e:
             # Catch other potential exceptions during send_message
             app.logger.error(f"Attempt {attempt + 1} failed with non-timeout error: {str(e)}")
             return {"target": "ERROR", "stance": f"API Error during send: {str(e)}"}
    # --- End Retry Logic --- #
    
    # Proceed only if response was successful
    if response is None:
         # Should have been handled by exception returns, but as a safeguard:
         return {"target": "ERROR", "stance": f"Failed to get response after retries. Last error: {str(last_exception)}"}
    
    try:
        # Try to parse the response as JSON
        response_text = response.text.strip()
        
        # Extract JSON if it's embedded in text (sometimes the model adds extra text)
        if not response_text.startswith('{'):
            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
                
        # Parse and validate the JSON
        parsed_response = json.loads(response_text)
        
        # Ensure required fields are present
        if 'target' not in parsed_response or 'stance' not in parsed_response:
            return {
                "error": "Invalid response format from model",
                "raw_response": response.text
            }
            
        # Validate target length (1-4 words)
        target_words = parsed_response['target'].split()
        if len(target_words) > 4:
            # Attempt to use the first 4 words, but handle if target is invalid
            try:
                parsed_response['target'] = ' '.join(target_words[:4])
            except Exception:
                parsed_response['target'] = "TARGET_PARSE_ERROR"
            
        # Normalize stance to uppercase
        parsed_response['stance'] = parsed_response['stance'].upper()
        
        # Validate stance value
        valid_stances = ["FAVOR", "AGAINST", "NEUTRAL"]
        if parsed_response['stance'] not in valid_stances:
            parsed_response['stance'] = "NEUTRAL"
        
        return parsed_response
        
    except json.JSONDecodeError:
        # If the response isn't valid JSON, return the raw text with error indication
        return {
            "target": "ERROR", 
            "stance": "Model response not JSON", 
            "raw_response": response.text
        }
    except Exception as e:
        # Catch any other exceptions during processing
        return {"target": "ERROR", "stance": f"Processing error: {str(e)}"}

def process_csv_background(task_id, file_stream_data, filename, text_column, gt_target_column, gt_stance_column, start_row):
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
            
            analysis_result = analyze_stance(input_text)
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
    result = analyze_stance(input_text)
    
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
    """Reads and returns the content of an evaluation result CSV."""
    safe_filename = secure_filename(filename)
    file_path = os.path.join(EVALUATION_DIR, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Evaluation file not found"}), 404
        
    try:
        df = pd.read_csv(file_path)
        json_string = df.to_json(orient='records', date_format='iso')
        results_data = json.loads(json_string)
        return jsonify({"data": results_data})
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
        args=(task_id, file_stream_data, file.filename, text_column, gt_target_column, gt_stance_column, start_row)
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
            app.logger         # Pass the logger instance
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
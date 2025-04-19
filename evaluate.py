import os
import json
import pandas as pd
import google.generativeai as genai
import google.api_core.exceptions
import time
import datetime
from werkzeug.utils import secure_filename
import requests 



def evaluate_results_background(task_id, analysis_filename, results_dir, evaluation_dir, task_queue, stop_event, logger, api_choice):
    
    
    q = task_queue 
    evaluation_results_list = []
    analysis_file_path = os.path.join(results_dir, secure_filename(analysis_filename))
    status = 'starting' 

    max_retries = 3
    base_delay = 1
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_api_url = "https://api.deepseek.com/chat/completions"
    deepseek_headers = {
        'Authorization': f'Bearer {deepseek_api_key}',
        'Content-Type': 'application/json'
    }
    gemini_model = None
    if api_choice == 'gemini':
        try:
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        except Exception as config_err:
             q.put(json.dumps({"type": "error", "message": f"Failed to configure Gemini: {config_err}"}))
             q.put("__ERROR__")
             return 
             
    elif api_choice == 'deepseek' and not deepseek_api_key:
         q.put(json.dumps({"type": "error", "message": "DeepSeek API Key not configured."}))
         q.put("__ERROR__")
         return 
    

    try:
        if not os.path.exists(analysis_file_path):
            raise FileNotFoundError(f"Analysis results file not found: {analysis_filename}")
        
        df_analysis = pd.read_csv(analysis_file_path)
        status = 'processing'
        total_rows = len(df_analysis)
        q.put(json.dumps({"type": "start", "total_rows": total_rows}))
        
        required_cols = ['Predicted Target', 'Predicted Stance', 'Ground Truth Target', 'Ground Truth Stance', 'Original Text']
        if not all(col in df_analysis.columns for col in required_cols):
            raise ValueError(f"Input CSV missing required columns. Need: {required_cols}")

        request_count = 0 
        pause_interval = 15 
        pause_duration = 60

        for index, row in df_analysis.iterrows():
            if stop_event.is_set():
                status = 'stopped'
                q.put(json.dumps({"type": "status", "message": f"Evaluation stopped by user at row {index + 1}."}))
                break

            pred_target = str(row['Predicted Target'])
            gt_target = str(row['Ground Truth Target'])
            pred_stance = str(row['Predicted Stance'])
            gt_stance = str(row['Ground Truth Stance'])
            original_text = str(row['Original Text'])
            
            q.put(json.dumps({
                "type": "progress", 
                "row": index + 1, 
                "total": total_rows,
                "pred_t": pred_target[:30] + '...' if len(pred_target)>30 else pred_target,
                "gt_t": gt_target[:30] + '...' if len(gt_target)>30 else gt_target,
                "pred_s": pred_stance,
                "gt_s": gt_stance
            }))
            
            # --- Single API Call Logic --- #
            target_match_pct = -9 # Default error code before call
            stance_match_pct = -9
            llm_error = None
            
            # Construct the single prompt asking for JSON output
            combined_prompt = f"""
            Evaluate the similarity between predicted and ground truth values for a stance detection task.
            Predicted Target: '{pred_target}'
            Ground Truth Target: '{gt_target}'
            Predicted Stance: '{pred_stance}'
            Ground Truth Stance: '{gt_stance}'
            
            Provide two scores:
            1. target_match_pct: Semantic similarity between targets (0-100).
            2. stance_match_pct: Similarity between stances (0-100). Consider FAVOR/AGAINST opposites (0), NEUTRAL vs FAVOR/AGAINST partial match (e.g., 50), identical match 100.
            
            Respond ONLY with a valid JSON object in the format:
            {{"target_match_pct": <integer_score>, "stance_match_pct": <integer_score>}}
            """

            last_exception = None
            for attempt in range(max_retries):
                try:
                    response_json = None
                    if api_choice == 'gemini':
                        # Gemini call
                        response = gemini_model.generate_content(combined_prompt)
                        response_text = response.text.strip()
                        # Extract JSON (Gemini might add backticks or other text)
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif response_text.startswith('{') and response_text.endswith('}'):
                             pass # Looks like JSON
                        else:
                             # Attempt to find JSON within potentially messy output
                             import re
                             match = re.search(r'({.*})', response_text, re.DOTALL)
                             if match:
                                 response_text = match.group(1)
                             else:
                                 raise ValueError("Could not extract JSON from Gemini response.")
                        response_json = json.loads(response_text)

                    elif api_choice == 'deepseek':
                        # DeepSeek call
                        payload = {
                            "model": "deepseek-chat", 
                            "messages": [{"role": "user", "content": combined_prompt}],
                            "stream": False,
                            "response_format": { "type": "json_object" }
                        }
                        response = requests.post(deepseek_api_url, headers=deepseek_headers, json=payload, timeout=60)
                        response.raise_for_status() 
                        response_data = response.json()
                        if 'error' in response_data:
                            raise Exception(f"DeepSeek API Error: {response_data['error']}")
                        message_content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        if not message_content:
                             raise ValueError("DeepSeek response empty.")
                        response_json = json.loads(message_content)

                    # Parse and validate the JSON response
                    if response_json and 'target_match_pct' in response_json and 'stance_match_pct' in response_json:
                        target_match_pct = int(response_json['target_match_pct'])
                        stance_match_pct = int(response_json['stance_match_pct'])
                        # Clamp values just in case
                        target_match_pct = max(0, min(100, target_match_pct))
                        stance_match_pct = max(0, min(100, stance_match_pct))
                        llm_error = None # Success
                        break # Exit retry loop on success
                    else:
                        raise ValueError("Parsed JSON response missing required keys.")

                # --- Exception Handling within Retry Loop --- #
                except (google.api_core.exceptions.DeadlineExceeded, requests.exceptions.Timeout) as e:
                    last_exception = e; error_code = -2; error_msg = "Timeout"
                except google.api_core.exceptions.ResourceExhausted as e:
                    last_exception = e; error_code = -6; error_msg = "Rate Limit/Quota"
                except requests.exceptions.HTTPError as e:
                     last_exception = e; error_code = -7; error_msg = f"HTTP Error {e.response.status_code}"
                     # Decide if retryable based on status code for HTTP errors
                     if e.response.status_code not in [429, 500, 503]:
                         llm_error = f"API Call Failed: {error_msg}"
                         break # Don't retry non-retryable HTTP errors
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    last_exception = e; error_code = -3; error_msg = "Response Parsing"
                    # Don't usually retry parsing errors, exit loop
                    llm_error = f"API Call Failed: {error_msg} ({e})"
                    break 
                except Exception as e:
                     last_exception = e; error_code = -4; error_msg = "General/Unknown"
                     # Break on unknown errors too to avoid infinite loops on persistent issues
                     llm_error = f"API Call Failed: {error_msg} ({e})"
                     break 
                
                # If we got here, it was a retryable error
                logger.warning(f"Evaluation Row {index+1} Attempt {attempt+1} failed: {error_msg}. Retrying in {base_delay * (2**attempt)}s...")
                if attempt < max_retries - 1:
                    # Check stop event during sleep
                    sleep_start = time.time()
                    while time.time() - sleep_start < (base_delay * (2**attempt)):
                         if stop_event.is_set(): break
                         time.sleep(0.1)
                    if stop_event.is_set(): break # Break outer loop if stopped during sleep
                else:
                    logger.error(f"Evaluation Row {index+1} failed after {max_retries} retries ({error_msg}). Last ex: {last_exception}")
                    target_match_pct = error_code # Use error code
                    stance_match_pct = error_code
                    llm_error = f"API Call Failed after retries: {error_msg}"
            # --- End Retry Loop --- #
            
            # If loop broken by stop event
            if stop_event.is_set():
                 status = 'stopped'
                 q.put(json.dumps({"type": "status", "message": f"Evaluation stopped by user at row {index + 1}."}))
                 break
                 
            # --- End Single API Call --- # 
            
            request_count += 1 # Increment request count *after* successful/failed attempt for the row
            
            # Handle case where loop finished but error occurred (e.g., non-retryable or parsing)
            if llm_error and target_match_pct >= -9:
                 target_match_pct = error_code if 'error_code' in locals() else -4
                 stance_match_pct = error_code if 'error_code' in locals() else -4
                 
            q.put(json.dumps({
                "type": "result", 
                "row": index + 1, 
                "target_match": target_match_pct if target_match_pct >= 0 else f"Error ({target_match_pct})",
                "stance_match": stance_match_pct if stance_match_pct >= 0 else f"Error ({stance_match_pct})"
            }))
            
            current_row_result = row.to_dict()
            current_row_result['Target Match Pct'] = target_match_pct
            current_row_result['Stance Match Pct'] = stance_match_pct
            evaluation_results_list.append(current_row_result)

            # Rate Limiting Pause (now based on rows/single calls)
            if request_count % pause_interval == 0 and index + 1 < total_rows: 
                try:
                    pause_msg = f"Pausing evaluation for {pause_duration}s after processing row {index+1} ({request_count} API calls)..."
                    q.put(json.dumps({"type": "status", "message": pause_msg}))
                    # Sleep allowing stop check
                    sleep_start = time.time()
                    while time.time() - sleep_start < pause_duration:
                        if stop_event.is_set(): break
                        time.sleep(0.1)
                        
                    if stop_event.is_set():
                        q.put(json.dumps({"type": "status", "message": "Pause interrupted by stop signal."}))
                        break 
                    else: 
                         q.put(json.dumps({"type": "status", "message": "Resuming evaluation..."}))
                except Exception as sleep_err:
                     q.put(json.dumps({"type": "error", "message": f"Error during pause: {str(sleep_err)}"}))
            
        else: # Loop completed normally
            status = 'completed'
            q.put(json.dumps({"type": "status", "message": "Evaluation completed successfully."}))

    except Exception as e:
        status = 'error'
        error_message = f"Error during evaluation: {str(e)}"
        q.put(json.dumps({"type": "error", "message": error_message}))
        logger.error(f"Task {task_id} evaluation error: {error_message}")

    finally:
        if evaluation_results_list:
            results_df = pd.DataFrame(evaluation_results_list)
            # We don't store results_df in active_tasks anymore here, as it's managed in app.py
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            status_suffix = "stopped" if status == 'stopped' else "completed"
            base_name = secure_filename(analysis_filename.rsplit('.csv', 1)[0])
            save_filename = f"{base_name}_evaluation_{timestamp}_{status_suffix}.csv"
            save_path = os.path.join(evaluation_dir, save_filename) 
            try:
                results_df.to_csv(save_path, index=False)
                q.put(json.dumps({"type": "saved", "filename": save_filename}))
            except Exception as e:
                 save_error = f"Failed to save evaluation results file {save_filename}: {str(e)}"
                 q.put(json.dumps({"type": "error", "message": save_error}))
                 logger.error(f"Task {task_id} evaluation save error: {save_error}")
        else:
             q.put(json.dumps({"type": "status", "message": "No evaluation results generated."}))
        
        q.put("__DONE__" if status != 'error' else "__ERROR__")
        print(f"Evaluation background task {task_id} finished with status: {status}") 
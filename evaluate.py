import os
import json
import pandas as pd
import google.generativeai as genai
import google.api_core.exceptions
import time
import datetime
from werkzeug.utils import secure_filename

# --- Evaluation Helper --- #

def get_evaluation_percentage(prompt_text, logger):
    """Calls Gemini API with a specific prompt asking for a percentage (0-100) 
       and parses the integer response. Includes retry logic."""
    if not prompt_text or not isinstance(prompt_text, str):
        return -1 # Indicate error

    # Consider making the model configurable if needed
    model = genai.GenerativeModel('gemini-1.5-pro') 
    
    max_retries = 3
    base_delay = 1
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Note: Ensure API key is configured in the main app context
            response = model.generate_content(prompt_text)
            response_text = response.text.strip()
            cleaned_text = "".join(filter(str.isdigit, response_text))
            if not cleaned_text:
                 raise ValueError("LLM response did not contain digits for percentage.")
            percentage = int(cleaned_text)
            if 0 <= percentage <= 100:
                return percentage
            else:
                 logger.warning(f"LLM percentage out of range ({percentage}). Clamping to 0-100.")
                 return max(0, min(100, percentage))

        except google.api_core.exceptions.DeadlineExceeded as e:
            last_exception = e
            logger.warning(f"Evaluation Attempt {attempt + 1} failed: DeadlineExceeded. Retrying in {base_delay * (2**attempt)}s...")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                 logger.error(f"Evaluation failed after {max_retries} retries (DeadlineExceeded).")
                 return -2 # Specific error code for timeout
        except google.api_core.exceptions.ResourceExhausted as e:
            last_exception = e
            logger.warning(f"Evaluation Attempt {attempt + 1} failed: Rate Limit/Quota (429). Retrying in {base_delay * (2**attempt)}s...")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                 logger.error(f"Evaluation failed after {max_retries} retries (Rate Limit/Quota).")
                 return -6 # Specific error code for rate limit
        except ValueError as ve:
             last_exception = ve
             logger.warning(f"Evaluation Attempt {attempt + 1} failed: ValueError ({ve}). Retrying in {base_delay * (2**attempt)}s...")
             if attempt < max_retries - 1:
                 time.sleep(base_delay * (2**attempt))
             else:
                 logger.error(f"Evaluation failed after {max_retries} retries (ValueError).")
                 return -3 # Specific error code for parsing value error
        except Exception as e:
             logger.error(f"Evaluation Attempt {attempt + 1} failed with non-timeout error: {str(e)}")
             return -4 # General error

    logger.error(f"Evaluation failed unexpectedly after retries. Last exception: {last_exception}")
    return -5 

# --- Background Evaluation Worker --- #

def evaluate_results_background(task_id, analysis_filename, results_dir, evaluation_dir, task_queue, stop_event, logger):
    """Background task to evaluate analysis results."""
    
    q = task_queue # Use passed queue
    evaluation_results_list = []
    analysis_file_path = os.path.join(results_dir, secure_filename(analysis_filename))
    status = 'starting' # Local status tracking

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
            
            # Call LLM for Target Match
            target_prompt = f"On a scale of 0 to 100, how semantically similar are these two short phrases intended as the target of a stance? Phrase 1: '{pred_target}'. Phrase 2: '{gt_target}'. Respond ONLY with the integer percentage." 
            target_match_pct = get_evaluation_percentage(target_prompt, logger)
            request_count += 1

            # Call LLM for Stance Match
            stance_match_pct = -1 
            if gt_stance != "N/A" and pred_stance != "ERROR":
                 stance_prompt = f"On a scale of 0 to 100, how well does the predicted stance '{pred_stance}' match the ground truth stance '{gt_stance}'? Consider FAVOR/AGAINST opposite (0% match) and NEUTRAL having partial match (e.g., 50%) with FAVOR/AGAINST. Respond ONLY with the integer percentage." 
                 stance_match_pct = get_evaluation_percentage(stance_prompt, logger)
                 request_count += 1
            else:
                 stance_match_pct = 0
            
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

            # Rate Limiting Pause
            if request_count % pause_interval == 0 and index + 1 < total_rows: 
                try:
                    pause_msg = f"Pausing evaluation for {pause_duration}s after {request_count} API calls..."
                    q.put(json.dumps({"type": "status", "message": pause_msg}))
                    for _ in range(pause_duration):
                        if stop_event.is_set():
                            q.put(json.dumps({"type": "status", "message": "Pause interrupted by stop signal."}))
                            break 
                        time.sleep(1)
                    else: 
                         q.put(json.dumps({"type": "status", "message": "Resuming evaluation..."}))
                         continue 
                    break
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
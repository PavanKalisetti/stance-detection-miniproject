import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd  # Added for CSV/Excel handling
import os            # Added for directory operations
import math          # Added for checking NaN

# --- Constants ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
PREDICTED_COL = "Predicted Target"
GROUND_TRUTH_COL = "Ground Truth Target"
SIMILARITY_COL = "target_match"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. Load Base BERTweet Model ---
# We use the base model for generating embeddings, not a fine-tuned one.
model_name = "vinai/bertweet-base"
print(f"Loading base BERTweet model: {model_name}...")

try:
    # Load tokenizer and model
    # BERTweet might have specific normalization needs, check documentation if results seem off
    tokenizer = AutoTokenizer.from_pretrained(model_name, normalize=True)
    model = AutoModel.from_pretrained(model_name)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"BERTweet base model loaded successfully on {device}.")

    # --- 3. Function to Get Embeddings ---
    def get_embedding(text, model, tokenizer, device):
        """Generates a single vector embedding for the input text."""
        # Handle potential NaN or non-string inputs explicitly
        if not isinstance(text, str) or text.strip() == "":
             # print(f"Warning: Invalid input text for embedding: '{text}'. Returning None.")
             return None # Return None for invalid/empty text

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs (no gradients needed)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the mean of the last hidden states as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        return embedding

    # --- 4. Process Data Files ---
    print(f"\nProcessing files in '{DATA_DIR}' directory...")

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".csv"):
            input_filepath = os.path.join(DATA_DIR, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_similarity.csv"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            print(f"\nProcessing file: {filename}...")

            try:
                df = pd.read_csv(input_filepath)

                # Check if required columns exist
                if PREDICTED_COL not in df.columns or GROUND_TRUTH_COL not in df.columns:
                    print(f"  Warning: Skipping file {filename}. Missing required columns: '{PREDICTED_COL}' or '{GROUND_TRUTH_COL}'.")
                    continue

                similarity_scores = []
                processed_rows = 0
                skipped_rows = 0
                # Lists to store labels for overall classification metrics
                y_true_for_metrics = []
                y_pred_for_metrics = []

                # Iterate through rows and calculate similarity
                for index, row in df.iterrows():
                    pred_target = row[PREDICTED_COL]
                    gt_target = row[GROUND_TRUTH_COL]

                    # Ensure targets are valid strings before getting embeddings
                    if pd.isna(pred_target) or not isinstance(pred_target, str) or pred_target.strip() == "" or \
                       pd.isna(gt_target) or not isinstance(gt_target, str) or gt_target.strip() == "":
                        # print(f"  Skipping row {index+2}: Invalid or empty target(s) ('{pred_target}', '{gt_target}').")
                        similarity_scores.append(np.nan) # Append NaN for skipped rows
                        skipped_rows += 1
                        continue

                    # Add valid labels to lists for overall metrics calculation
                    y_true_for_metrics.append(gt_target)
                    y_pred_for_metrics.append(pred_target)

                    # Get embeddings
                    pred_embedding = get_embedding(pred_target, model, tokenizer, device)
                    gt_embedding = get_embedding(gt_target, model, tokenizer, device)

                    # Calculate cosine similarity if embeddings are valid
                    if pred_embedding is not None and gt_embedding is not None:
                        # Handle potential single-element arrays if embedding failed gracefully before
                        if pred_embedding.ndim == 1: pred_embedding = pred_embedding.reshape(1, -1)
                        if gt_embedding.ndim == 1: gt_embedding = gt_embedding.reshape(1, -1)

                        score = cosine_similarity(pred_embedding, gt_embedding)[0][0]
                        similarity_scores.append(score)
                        processed_rows += 1
                    else:
                        # print(f"  Skipping row {index+2}: Could not generate embedding for one or both targets.")
                        similarity_scores.append(np.nan) # Append NaN if embedding failed
                        skipped_rows += 1

                # Add scores as a new column
                df[SIMILARITY_COL] = similarity_scores

                # Save to CSV instead of Excel
                df.to_csv(output_filepath, index=False)
                print(f"  Finished processing. Results saved to: {output_filepath}")
                print(f"  Processed rows: {processed_rows}, Skipped rows (empty/invalid targets): {skipped_rows}")

                # --- Calculate and Print Overall Classification Metrics ---
                if y_true_for_metrics and y_pred_for_metrics:
                    print("\n  --- Overall Classification Metrics for this file ---")
                    try:
                        # Determine unique labels present in the ground truth for the report
                        unique_labels = sorted(list(set(y_true_for_metrics) | set(y_pred_for_metrics)))

                        accuracy = accuracy_score(y_true_for_metrics, y_pred_for_metrics)
                        report = classification_report(y_true_for_metrics, y_pred_for_metrics, labels=unique_labels, zero_division=0)

                        print(f"  Accuracy: {accuracy:.4f}")
                        print("  Classification Report:")
                        # Indent the report for clarity
                        print("\n".join([f"    {line}" for line in report.splitlines()]))
                    except Exception as metric_e:
                        print(f"  Could not calculate classification metrics: {metric_e}")
                else:
                    print("  Skipping overall classification metrics (no valid label pairs found).")
                # --- End Metrics Calculation ---

            except FileNotFoundError:
                print(f"  Error: File not found: {input_filepath}")
            except pd.errors.EmptyDataError:
                print(f"  Warning: File is empty: {input_filepath}")
            except Exception as e:
                print(f"  An unexpected error occurred while processing {filename}: {e}")


except Exception as e:
    print(f"\nAn error occurred during model loading or setup: {e}")
    print("Please ensure the model name is correct, you have transformers, torch, scikit-learn, pandas, openpyxl, and numpy installed (`pip install transformers torch scikit-learn numpy pandas openpyxl`), and the necessary model files can be downloaded.")

print("\nScript finished.") 
import pandas as pd
import os

# Define the directory containing the CSV files
output_dir = "output"

# List of CSV files identified previously
csv_files = [
    "subtaskA_noun_phrases_finetune_model_similarity.csv",
    "C19_test_base_model_similarity.csv",
    "subtaskA-mixed_finetuned_model_similarity.csv",
    "c19_finetune_model_results_similarity.csv",
    "subtask_A_noun_phrases_base_model_similarity.csv",
    "subtaskA-mixed_basemodel_similarity.csv"
]

print("BERT-TWEET MODEL AVERAGE SIMILARITY OF TARGET MATCH: ")

# Loop through each file
for filename in csv_files:
    file_path = os.path.join(output_dir, filename)
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the 'target_match' column exists
        if 'target_match' in df.columns:
            # Convert the column to numeric, coercing errors to NaN
            numeric_matches = pd.to_numeric(df['target_match'], errors='coerce')

            # Drop NaN values and calculate the mean
            average_match = numeric_matches.dropna().mean()

            print(f"- {filename}: {average_match:.4f}")
        else:
            print(f"- {filename}: 'target_match' column not found.")

    except FileNotFoundError:
        print(f"- {filename}: File not found.")
    except pd.errors.EmptyDataError:
        print(f"- {filename}: File is empty.")
    except Exception as e:
        print(f"- {filename}: An error occurred - {e}")

print("\nCalculation complete.") 
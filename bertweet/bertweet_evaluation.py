import torch
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Input Data ---
# Replace these with your actual data
texts = [
    "This new policy is worst!",
    "I'm really concerned about the impact on local communities.",
    "The report seems neutral on the main issue.",
    "Strongly disagree with the proposed changes.",
]

# Predictions from your model (must correspond to the texts above)
your_model_predictions = ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE"]

# Ground truth labels (must correspond to the texts above)
ground_truth_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE"]

# Define the possible labels for your task
# Adjust these labels based on your specific problem (e.g., 'FAVOR', 'AGAINST', 'NONE')
task_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
# --- --- --- --- --- ---

# --- 2. Load Fine-Tuned BERTweet Model ---
# Using a BERTweet model fine-tuned for sentiment analysis as an example.
# Find models: https://huggingface.co/models?search=bertweet
# You might need to replace 'finiteautomata/bertweet-base-sentiment-analysis'
# with a model fine-tuned specifically for your task and labels.
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
print(f"Loading BERTweet model: {model_name}...")

try:
    # Use pipeline for easy prediction
    # device=0 for GPU if available, -1 for CPU
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=True), # BERTweet often requires its specific tokenizer setup
        device=0 if torch.cuda.is_available() else -1
    )
    print("BERTweet model loaded successfully.")

    # --- 3. Get Predictions from BERTweet ---
    bertweet_raw_predictions = classifier(texts)
    print("\nRaw predictions from BERTweet:")
    print(bertweet_raw_predictions)

    # --- 4. Extract BERTweet Predicted Labels ---
    # The output labels from the pipeline might be different (e.g., 'POS', 'NEG', 'NEU').
    # You MUST map these to your task_labels ('POSITIVE', 'NEGATIVE', 'NEUTRAL').
    # *** This mapping is CRITICAL and depends entirely on the chosen model. ***
    # Inspect bertweet_raw_predictions to determine the correct mapping.
    label_map = {
        'POS': 'POSITIVE',
        'NEG': 'NEGATIVE',
        'NEU': 'NEUTRAL',
         # Add other labels if needed based on the model's output
    }

    bertweet_predicted_labels = []
    for pred in bertweet_raw_predictions:
        # Map the predicted label, default to a placeholder if not found
        mapped_label = label_map.get(pred['label'], 'UNKNOWN_BERTWEET_LABEL')
        bertweet_predicted_labels.append(mapped_label)

    print("\nMapped BERTweet predictions:")
    print(bertweet_predicted_labels)

    # --- 5. Evaluate Predictions ---
    print("\n--- Evaluation Report ---")

    print("\nEvaluating YOUR model's predictions:")
    print(f"Accuracy: {accuracy_score(ground_truth_labels, your_model_predictions):.4f}")
    print(classification_report(ground_truth_labels, your_model_predictions, labels=task_labels, zero_division=0))

    print("\nEvaluating BERTweet model's predictions:")
    # Check if BERTweet predictions could be mapped correctly
    if 'UNKNOWN_BERTWEET_LABEL' in bertweet_predicted_labels:
         print("Warning: Some BERTweet predictions could not be mapped to your task labels.")
         print("Please check the 'label_map' and the raw BERTweet output.")
         # Filter out unmapped predictions for calculation or handle appropriately
         valid_indices = [i for i, label in enumerate(bertweet_predicted_labels) if label != 'UNKNOWN_BERTWEET_LABEL']
         if valid_indices:
             filtered_gt = [ground_truth_labels[i] for i in valid_indices]
             filtered_bertweet = [bertweet_predicted_labels[i] for i in valid_indices]
             print(f"Accuracy (on mapped labels): {accuracy_score(filtered_gt, filtered_bertweet):.4f}")
             print(classification_report(filtered_gt, filtered_bertweet, labels=task_labels, zero_division=0))
         else:
              print("No valid mapped BERTweet predictions to evaluate.")

    else:
        print(f"Accuracy: {accuracy_score(ground_truth_labels, bertweet_predicted_labels):.4f}")
        print(classification_report(ground_truth_labels, bertweet_predicted_labels, labels=task_labels, zero_division=0))


except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure the model name is correct, you have transformers installed (`pip install transformers torch scikit-learn`), and the necessary model files can be downloaded.")

print("\nScript finished.") 
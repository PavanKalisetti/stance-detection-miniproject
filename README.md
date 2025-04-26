Stance Detection API with Gemini, DeepSeek, and Ollama
This API uses Google's Gemini, DeepSeek, or local Ollama models to detect the target and stance in text inputs for Open-Target Stance Detection (OTSD). The target is limited to 1-4 words, and the stance is classified as FAVOR, AGAINST, or NEUTRAL.
Setup

Clone this repository:git clone https://github.com/PavanKalisetti/stance-detection-miniproject.git


Install dependencies:pip install -r requirements.txt


Create a .env file and add your API keys and/or Ollama configuration:GEMINI_API_KEY=your_gemini_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2:1b




OLLAMA_URL and OLLAMA_MODEL are optional. If using Ollama, ensure your Ollama server is running locally and the model is pulled (see https://ollama.com/library for available models).

Running the API
Run the Flask app:
python app.py

The API will be available at http://localhost:5000
Usage
Web Interface
The web application provides two ways to analyze text:

Manual Input: Enter text directly in the text area or select from example texts.
CSV Upload: Upload a CSV file and select the text column and optional ground truth columns.

Model Selection:

Choose between Gemini, DeepSeek, or Ollama from the UI for both manual and CSV analysis.

CSV Requirements
When uploading a CSV file:

The file must contain a column with the text content to analyze (default: "text").
You can specify the column name in the UI.
Optionally, provide columns for ground truth target and stance for evaluation.

API Endpoints
Send a POST request to /detect_stance with a JSON body containing the text to analyze and the API/model to use:
curl -X POST http://localhost:5000/detect_stance \
  -H "Content-Type: application/json" \
  -d '{"text": "I believe climate change is a serious threat that requires immediate action.", "api_choice": "ollama"}'


api_choice can be gemini, deepseek, or ollama (default is gemini if not specified).

Response Format
The API returns a JSON object with the target and stance:
{
  "target": "climate change",
  "stance": "FAVOR"
}

The target will be 1-4 words that identify what the text is about. The stance will be one of: FAVOR, AGAINST, or NEUTRAL.

Ollama Support

To use local models, install and run Ollama (https://ollama.com/).
Pull a model (e.g., ollama pull llama2 or ollama pull qwen2.5:1.5b).
Ensure OLLAMA_URL and OLLAMA_MODEL are set in your .env file.
Select "Ollama" in the web UI or pass "api_choice": "ollama" in API requests.

Finetuning Ollama Models
You can finetune local Ollama models for improved OTSD performance using the provided notebook:

finetuning/Llama3_1_(8B)_Alpaca.ipynb

Finetuning Details:

Model: Llama 3.1 8B, finetuned using Low-Rank Adaptation (LoRA) with rank=16, alpha=16, targeting query, key, value, output, gate, up, and down projection layers.
Dataset: Combined TSE and VAST datasets (6,480 examples) for training, covering diverse topics and stances.
Optimization: Uses 4-bit quantization and Unsloth library for efficiency on NVIDIA T4 GPU.
Hyperparameters: Max sequence length=2048, batch size=2, gradient accumulation=4, learning rate=2e-4, 60 training steps.
How to Use:
Open the notebook in Jupyter or VS Code (Google Colab not recommended).
Run all cells in order (no manual edits needed).
Follow instructions in the notebook to save and use the finetuned model.



Finetuning enhances target identification and stance classification, especially for specialized domains like COVID-19.
Evaluation and Results
The finetuned Llama 3.1 8B model was evaluated on three datasets:

COVID-19: Specialized domain with pandemic-related texts.
EZStance Mixed: General-domain dataset with diverse topics.
EZStance Noun Phrase: Targets are noun phrases, posing linguistic challenges.

Key Results:

COVID-19: Finetuned model achieves 66.30% stance accuracy (vs. 45.28% base) and 52.47% target accuracy (vs. 25.39% base).
EZStance Mixed: Improved stance accuracy by 5.24% and target accuracy by 7.30%, with BERTweet semantic similarity rising from 0.7392 to 0.7779.
EZStance Noun Phrase: Target accuracy increased by 8.28%, but stance accuracy slightly declined, highlighting domain-specific challenges.
Evaluation Method: Semantic assessment by Gemini and DeepSeek-671B, averaged for robustness, with BERTweet for target similarity and Macro F1 for stance classification.

The finetuned model significantly outperforms the base model, particularly in specialized domains, validating the effectiveness of LoRA-based finetuning for OTSD.

Notes

For Gemini and DeepSeek, valid API keys are required.
Ollama requires a locally running server but no API key.
The finetuned model is not included in the repository due to size; follow the finetuning notebook to generate it.


For issues or contributions, please open an issue or pull request on GitHub.

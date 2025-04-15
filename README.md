# Stance Detection API with Gemini, DeepSeek, and Ollama

This API uses Google's Gemini, DeepSeek, or local Ollama models to detect the target and stance in text inputs. The target is limited to 1-4 words.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your API keys and/or Ollama configuration:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   OLLAMA_URL=http://localhost:11434/api/generate
   OLLAMA_MODEL=llama3.2:1b
   ```

- `OLLAMA_URL` and `OLLAMA_MODEL` are optional. If using Ollama, ensure your Ollama server is running locally and the model is pulled (see https://ollama.com/library for available models).

## Running the API

Run the Flask app:
```
python app.py
```

The API will be available at http://localhost:5000

## Usage

### Web Interface

The web application provides two ways to analyze text:

1. **Manual Input**: Enter text directly in the text area or select from example texts
2. **CSV Upload**: Upload a CSV file and select the text column and optional ground truth columns

**Model Selection:**
- You can choose between Gemini, DeepSeek, or Ollama from the UI for both manual and CSV analysis.

### CSV Requirements

When uploading a CSV file:
- The file must contain a column with the text content to analyze (default: "text").
- You can specify the column name in the UI.
- Optionally, provide columns for ground truth target and stance for evaluation.

### API Endpoints

Send a POST request to `/detect_stance` with a JSON body containing the text to analyze and the API/model to use:

```bash
curl -X POST http://localhost:5000/detect_stance \
  -H "Content-Type: application/json" \
  -d '{"text": "I believe climate change is a serious threat that requires immediate action.", "api_choice": "ollama"}'
```

- `api_choice` can be `gemini`, `deepseek`, or `ollama` (default is `gemini` if not specified).

### Response Format

The API returns a JSON object with the target and stance:

```json
{
  "target": "climate change",
  "stance": "FAVOR"
}
```

The target will be 1-4 words that identify what the text is about.
The stance will be one of: FAVOR, AGAINST, or NEUTRAL.

---

## Ollama Support
- To use local models, install and run Ollama (https://ollama.com/).
- Pull a model (e.g., `ollama pull llama2` or `ollama pull qwen2.5:1.5b`).
- Make sure `OLLAMA_URL` and `OLLAMA_MODEL` are set in your `.env` file.
- Select "Ollama" in the web UI or pass `"api_choice": "ollama"` in API requests.

## Finetuning Ollama Models

You can finetune local Ollama models for your stance detection tasks. An example notebook is provided:

- [`finetuning/Llama3_1_(8B)_Alpaca.ipynb`](finetuning/Llama3_1_(8B)_Alpaca.ipynb)

**How to use:**
1. Open the notebook in Jupyter or VS Code recommended (google collab).
2. Run all cells in order (no manual edits needed).
3. Follow any additional instructions in the notebook for saving or using your finetuned model.

This allows you to adapt local models to your own data and improve performance for your specific use case.

---

## Notes
- For Gemini and DeepSeek, you must provide valid API keys.
- Ollama does not require an API key but must be running locally or at the specified `OLLAMA_URL`.

---

For any issues or contributions, please open an issue or pull request.
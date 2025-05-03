# Stance Detection API

Detect the **target** and **stance** in text using Google Gemini, DeepSeek, or local Ollama models. Designed for Open-Target Stance Detection (OTSD), this API identifies a concise target (1-4 words) and classifies stance as `FAVOR`, `AGAINST`, or `NEUTRAL`.

---

## 🚀 Features
- **Multiple Model Support:** Gemini, DeepSeek, or local Ollama (Llama 3.2:1b and more)
- **Web UI & API:** Analyze text manually or via CSV upload
- **Finetuning:** Instructions for finetuning Ollama models for improved performance
- **Evaluation:** Benchmarked on COVID-19 and EZStance datasets

---

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PavanKalisetti/stance-detection-miniproject.git
   cd stance-detection-miniproject
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     OLLAMA_URL=http://localhost:11434/api/generate
     OLLAMA_MODEL=llama3.2:1b
     ```
   - `OLLAMA_URL` and `OLLAMA_MODEL` are optional. If using Ollama, ensure your server is running and the model is pulled ([see Ollama library](https://ollama.com/library)).

---

## 🏃 Running the API

Start the Flask app:
```bash
python app.py
```
The API will be available at [http://localhost:5000](http://localhost:5000)

---

## 💻 Usage

### Web Interface
- **Manual Input:** Enter text or select from examples.
- **CSV Upload:** Upload a CSV, select the text column, and (optionally) ground truth columns.
- **Model Selection:** Choose Gemini, DeepSeek, or Ollama in the UI.

### CSV Requirements
- Must contain a column with the text to analyze (default: `text`).
- You can specify the column name in the UI.
- Optionally, provide columns for ground truth target and stance for evaluation.

### API Endpoint
Send a POST request to `/detect_stance`:
```bash
curl -X POST http://localhost:5000/detect_stance \
  -H "Content-Type: application/json" \
  -d '{"text": "I believe climate change is a serious threat that requires immediate action.", "api_choice": "ollama"}'
```
- `api_choice`: `gemini`, `deepseek`, or `ollama` (default: `gemini`)

#### Example Response
```json
{
  "target": "climate change",
  "stance": "FAVOR"
}
```
- **Target:** 1-4 words summarizing the topic
- **Stance:** One of `FAVOR`, `AGAINST`, `NEUTRAL`

---

## 🦙 Ollama Support
- [Install Ollama](https://ollama.com/)
- Pull a model (e.g., `ollama pull llama2` or `ollama pull qwen2.5:1.5b`)
- Set `OLLAMA_URL` and `OLLAMA_MODEL` in your `.env`
- Select "Ollama" in the web UI or use `"api_choice": "ollama"` in API requests

---

## 🧑‍🔬 Finetuning Ollama Models

Finetune local Ollama models for improved OTSD performance:
- **Just run this notebook:** [`finetuning/Llama3_1_(8B)_Alpaca.ipynb`](finetuning/Llama3_1_(8B)_Alpaca.ipynb)

**Finetuning Details:**
- **Model:** Llama 3.1 8B, LoRA (rank=16, alpha=16)
- **Dataset:** Combined TSE and VAST (6,480 examples)
- **Optimization:** 4-bit quantization, Unsloth, NVIDIA T4 GPU
- **Hyperparameters:** Max seq len=2048, batch=2, grad accum=4, lr=2e-4, 60 steps

**How to Use:**
1. Open the notebook in Jupyter or VS Code (not Colab)
2. Run all cells in order
3. Follow instructions to save and use the finetuned model

> Finetuning improves target identification and stance classification, especially for specialized domains like COVID-19.

---

## 📊 Evaluation & Results

**Datasets:**
- COVID-19 (pandemic-related)
- EZStance Mixed (general topics)
- EZStance Noun Phrase (linguistic challenge)

**Key Results:**
- **COVID-19:** Finetuned model: 66.3% stance accuracy (vs. 45.3% base), 52.5% target accuracy (vs. 25.4% base)
- **EZStance Mixed:** +5.2% stance, +7.3% target accuracy, BERTweet similarity: 0.7779 (vs. 0.7392)
- **EZStance Noun Phrase:** +8.3% target accuracy, slight stance decline (domain-specific)

**Evaluation:**
- Semantic assessment by Gemini and DeepSeek-671B (averaged)
- BERTweet for target similarity
- Macro F1 for stance classification

> The finetuned model significantly outperforms the base, especially in specialized domains, validating LoRA-based finetuning for OTSD.

---

## 📝 Notes
- Gemini and DeepSeek require valid API keys
- Ollama requires a local server (no API key)
- The finetuned model is not included due to size; use the notebook to generate it

---

## 🤝 Contributing & Issues
For issues or contributions, please [open an issue or pull request](https://github.com/PavanKalisetti/stance-detection-miniproject/issues) on GitHub.

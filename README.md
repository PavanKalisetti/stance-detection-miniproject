# Stance Detection API with Gemini

This API uses Google's Gemini model to detect the target and stance in text inputs. The target is limited to 1-4 words.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Running the API

Run the Flask app:
```
python app.py
```

The API will be available at http://localhost:5000

## Usage

Send a POST request to `/detect_stance` with a JSON body containing the text to analyze:

```bash
curl -X POST http://localhost:5000/detect_stance \
  -H "Content-Type: application/json" \
  -d '{"text": "I believe climate change is a serious threat that requires immediate action."}'
```

### Response Format

The API returns a JSON object with the response from Gemini:

```json
{
  "target": "climate change",
  "stance": "FAVOR"
}
```

The target will be 1-4 words that identify what the text is about.
The stance will be one of: FAVOR, AGAINST, or NEUTRAL. 
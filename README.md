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

### Web Interface

The web application provides two ways to analyze text:

1. **Manual Input**: Enter text directly in the text area or select from example texts
2. **CSV Upload**: Upload a CSV file with a "post" column and select rows to analyze

### CSV Requirements

When uploading a CSV file:
- The file must contain a column named "post" with the text content to analyze
- After uploading, you can select any row to analyze the stance

### API Endpoints

Send a POST request to `/detect_stance` with a JSON body containing the text to analyze:

```bash
curl -X POST http://localhost:5000/detect_stance \
  -H "Content-Type: application/json" \
  -d '{"text": "I believe climate change is a serious threat that requires immediate action."}'
```

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
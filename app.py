from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_stance', methods=['POST'])
def detect_stance():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    input_text = data['text']
    
    # System prompt for stance detection with structured output
    system_prompt = """
    You are a stance detection system. Analyze the given text and identify:
    1. The target - what the text is discussing or taking a stance on. IMPORTANT: The target must be only 1-4 words maximum.
    2. The stance - whether the author is in FAVOR, AGAINST, or NEUTRAL toward the target
    
    You MUST respond ONLY with a valid JSON object in the following format:
    {
      "target": "identified target (1-4 words only)",
      "stance": "STANCE"
    }
    
    Where STANCE is one of: FAVOR, AGAINST, or NEUTRAL.
    Do not include any explanation or additional text in your response.
    """
    
    # Generate response from Gemini
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [system_prompt]
        },
        {
            "role": "model",
            "parts": ["I'll analyze text and respond with only a JSON object containing the target and stance."]
        }
    ])
    
    response = chat.send_message(input_text)
    
    try:
        # Try to parse the response as JSON
        response_text = response.text.strip()
        
        # Extract JSON if it's embedded in text (sometimes the model adds extra text)
        if not response_text.startswith('{'):
            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
                
        # Parse and validate the JSON
        parsed_response = json.loads(response_text)
        
        # Ensure required fields are present
        if 'target' not in parsed_response or 'stance' not in parsed_response:
            return jsonify({
                "error": "Invalid response format from model",
                "raw_response": response.text
            }), 500
            
        # Validate target length (1-4 words)
        target_words = parsed_response['target'].split()
        if len(target_words) > 4:
            # Truncate to first 4 words
            parsed_response['target'] = ' '.join(target_words[:4])
            
        # Normalize stance to uppercase
        parsed_response['stance'] = parsed_response['stance'].upper()
        
        # Validate stance value
        valid_stances = ["FAVOR", "AGAINST", "NEUTRAL"]
        if parsed_response['stance'] not in valid_stances:
            parsed_response['stance'] = "NEUTRAL"
        
        return jsonify(parsed_response)
        
    except json.JSONDecodeError:
        # If the response isn't valid JSON, return the raw text
        return jsonify({
            "error": "Could not parse JSON from model response",
            "raw_response": response.text
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
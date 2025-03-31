import requests
import json

def test_stance_detection(text):
    url = "http://localhost:5000/detect_stance"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "text": text
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test examples
    examples = [
        "I believe climate change is a serious threat that requires immediate action.",
        "The new tax policy won't help the economy at all and will only benefit the wealthy.",
        "While artificial intelligence has potential benefits, it also poses significant risks that need to be addressed.",
        "The latest smartphone release has some interesting features, but I'm not sure if it's worth upgrading."
    ]
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Text: {example}")
        test_stance_detection(example)
        print("-" * 50) 
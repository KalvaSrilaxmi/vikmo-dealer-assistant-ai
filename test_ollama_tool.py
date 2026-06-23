import os
import sys
import json
import requests

# Ensure local imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from assistant.llm import translate_history_to_openai, OPENAI_TOOLS
from assistant.agent import SYSTEM_INSTRUCTION

def test_ollama():
    contents = [
        {
            "role": "user",
            "parts": ["Do you have TVS brake pads for the Bajaj Pulsar 150?"]
        }
    ]
    messages = translate_history_to_openai(contents, SYSTEM_INSTRUCTION, arguments_as_object=True)
    
    payload = {
        "model": "llama3.1",
        "messages": messages,
        "options": {"temperature": 0.0},
        "stream": False,
        "tools": OPENAI_TOOLS
    }
    
    url = "http://localhost:11434/api/chat"
    print("Sending payload to Ollama...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print("Status Code:", response.status_code)
        res_json = response.json()
        print("\nResponse Message:")
        print(json.dumps(res_json.get("message"), indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_ollama()

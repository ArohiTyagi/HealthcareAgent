import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3",
    "prompt": "Hello",
    "stream": False
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Ollama is working!")
        print("Response:", response.json().get("response"))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to Ollama. Is it running on localhost:11434?")
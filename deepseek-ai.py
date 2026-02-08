import requests
import os

# Specify the full path
directory = "/home/tien/"  
filename = "api-openrouter.txt"
API_PATH = os.path.join(directory, filename)

# Method 1: Basic file reading
def load_api_key_simple(filepath):
    with open(filepath, 'r') as file:
        api_key = file.read().strip()  # .strip() removes whitespace/newlines
    return api_key

# Usage
api_key = load_api_key_simple(API_PATH)


# Your OpenRouter API key (replace with your actual key)
API_KEY = api_key
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Our question to DeepSeek
data = {
    "model": "tngtech/deepseek-r1t2-chimera:free",
    "messages": [{"role": "user", "content": "What's the best way to learn programming?"}]
}

# Send the request
response = requests.post(API_URL, json=data, headers=headers)

# Check if it worked
if response.status_code == 200:
    # Extract and print just the AI's response text
    ai_message = response.json()['choices'][0]['message']['content']
    print(f"DeepSeek says: {ai_message}")
else:
    print(f"Oops! Something went wrong. Status code: {response.status_code}")
    print(f"Error details: {response.text}")

from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Set device to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Set up pipeline for text generation
pipe = pipeline(
    "text-generation", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
    device_map=device
)

message = [
    {
        "role": "system",
        "content": """
        You are a friendly chatbot named Chatty
        """
    },
    {
        "role": "user", 
        "content": """
        Please introduce yourself and add
        'how can I help you today?' at
        the end of the response
        """
    }
]

def ask_question(msg):
    # Generate response from deepseek model
    message[1]["content"] = msg
    response = pipe(
        message, 
        max_new_tokens=2048
    )[0]['generated_text'][-1]["content"].split("</think>")
    
    think = response[0].strip().replace("<think>", "")
    say = response[1].strip()
    print(say)
    # Return both think and say parts as a dictionary
    return {
        "think": think,
        "say": say
    }

@app.route('/query', methods=['POST'])
def query():
    # Extract the message from the request
    data = request.get_json()
    message = data.get('message')
    print(message)

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Generate response using the ask_question function
    result = ask_question(message)

    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)

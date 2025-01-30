import os
import requests

async def call_lmstudio_model(messages, objective, model):
    """
    Call the LMStudio model with the given messages and objective.
    
    :param messages: List of conversation messages.
    :param objective: The overall goal of the conversation.
    :param model: Model identifier (not used in LMStudio but kept for consistency).
    :return: The response from LMStudio.
    """
    api_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:8000/api/v1/completions")
    api_key = os.getenv("LMSTUDIO_API_KEY")

    if not api_key:
        raise ValueError("API key for LMStudio is not set. Please provide it explicitly or set LMSTUDIO_API_KEY as an environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": objective + "\n" + "\n".join(msg["content"] for msg in messages),
        "max_tokens": 256,
        "temperature": 0.7
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("text", "")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to LMStudio API: {e}")

# Example Usage
if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "What is the capital of France?"}]
    test_objective = "Answer the user's question."
    
    try:
        response = call_lmstudio_model(test_messages, test_objective, "lmstudio")
        print("Model Response:", response)
    except Exception as e:
        print("Error:", e)

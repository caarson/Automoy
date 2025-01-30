import os
import requests

async def call_lmstudio_model(messages, objective, model):
    """
    Call the LMStudio model with the given messages and objective.
    
    :param messages: List of conversation messages.
    :param objective: The overall goal of the conversation.
    :param model: Model identifier (e.g., 'llama-3.2-1b-instruct').
    :return: The response text from LM Studio.
    """
    api_url = "http://10.27.167.59:1234/v1/completions"  # Base LM Studio URL
    headers = {"Content-Type": "application/json"}

    # Combine objective and messages into a single prompt
    prompt = f"{objective}\n\n" + "\n".join(msg["content"] for msg in messages)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 10000,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to LMStudio API: {e}")

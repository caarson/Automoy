import os
import requests
import asyncio
from operate.config import Config  # Corrected import path

async def call_lmstudio_model(messages, objective, model):
    config = Config()
    api_source, api_value = config.get_api_source()  # Get the proper API source

    if api_source == "lmstudio":
        api_url = api_value  # Use LMStudio API URL
    elif api_source == "openai":
        raise RuntimeError("LMStudio API is not available, and OpenAI API is not implemented in this function.")
    else:
        raise RuntimeError("No valid API source found.")

    headers = {"Content-Type": "application/json"}
    prompt = f"{objective}\n\n" + "\n".join(msg["content"] for msg in messages)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to LMStudio API: {e}")

async def test_lmstudio(model):
    test_messages = [{"role": "user", "content": "tell me about google"}]
    test_objective = "Answer the user's question."

    try:
        response = await call_lmstudio_model(test_messages, test_objective, model)
        print("Test Response:", response)
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    asyncio.run(test_lmstudio(model="deepseek-r1-distill-llama-8b"))

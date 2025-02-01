import os
import requests
import asyncio

async def call_lmstudio_model(messages, objective, model):
    api_url = "http://10.27.212.231:1234/v1/completions"
    headers = {"Content-Type": "application/json"}

    prompt = f"{objective}\n\n" + "\n".join(msg["content"] for msg in messages)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 500,  # Limit tokens
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

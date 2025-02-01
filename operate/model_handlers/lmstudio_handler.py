import os
import requests
import asyncio
import re
import json

def extract_json_from_text(content):
    """
    Extracts JSON from response text using regex.
    Returns an empty list if no JSON is found.
    """
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
        return json_text.strip()
    else:
        print("[ERROR] No JSON block detected in response.")
        return "[]"

def fix_json_format(raw_response):
    """
    Cleans up raw response to ensure it's valid JSON before parsing.
    """
    try:
        return json.loads(raw_response)  # Try parsing first
    except json.JSONDecodeError:
        print("[WARNING] Response has incorrect JSON formatting. Attempting auto-fix...")
        
        # Remove invalid characters and fix common formatting issues
        cleaned_response = re.sub(r'[^a-zA-Z0-9:{}\[\],"\'.\s]', '', raw_response)
        cleaned_response = re.sub(r',?\s*{}\s*', '', cleaned_response)
        cleaned_response = re.sub(r'"done"\s*:\s*{(.*?)\s*}', r'"done": {"summary": "\1"}', cleaned_response)
        cleaned_response = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', cleaned_response)
        
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON Decode Failed After Fixing: {e}")
            return []

async def call_lmstudio_model(messages, objective, model):
    """
    Calls the LMStudio model with the given messages and objective.
    Parses and returns JSON if the response contains a JSON block.

    :param messages: List of conversation messages.
    :param objective: The overall goal of the conversation.
    :param model: Model identifier (e.g., 'llama-3.2-1b-instruct').
    :return: Parsed JSON response or raw text if parsing fails.
    """
    api_url = "http://10.27.212.231:1234/v1/completions"
    headers = {"Content-Type": "application/json"}

    # Combine objective and messages into a single prompt
    prompt = f"{objective}\n\n" + "\n".join(msg["content"] for msg in messages)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100000000,  # Limit tokens
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        raw_text = response.json().get("choices", [{}])[0].get("text", "").strip()
        
        # Attempt to extract and parse JSON
        json_text = extract_json_from_text(raw_text)
        parsed_json = fix_json_format(json_text)
        return parsed_json
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to connect to LMStudio API: {e}")

async def test_lmstudio(model):
    """
    Test function for calling the LM Studio model.
    """
    test_messages = [{"role": "user", "content": "tell me about google"}]
    test_objective = "Answer the user's question."

    try:
        response = await call_lmstudio_model(test_messages, test_objective, model)
        print("Test Response:", response)
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    asyncio.run(test_lmstudio(model="deepseek-r1-distill-llama-8b"))

import os
import sys
import requests
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from operate.config import Config  # Corrected import path

# Ensure all messages have "role" and "content" fields
# ✅ Helper function to format OCR & YOLO data into a readable string
def format_preprocessed_data(data):
    """
    Converts preprocessed OCR and YOLO data into a clean string format.
    Ensures it's properly formatted for LMStudio without breaking JSON parsing.
    """
    if not isinstance(data, dict):
        return "No valid preprocessed data."

    formatted_string = "\n".join(
        [f"{key}: {', '.join(map(str, value))}" for key, value in data.items()]
    )
    
    return f"Preprocessed data with OCR and YOLO:\n{formatted_string}"

# ✅ Ensures messages follow OpenAI's API format
def format_messages(messages):
    formatted = []
    for msg in messages:
        if "role" in msg and "content" in msg:
            if msg["content"] is None or not isinstance(msg["content"], str):
                print(f"[WARNING] Skipping malformed message: {msg}")  # ✅ Debugging
                continue  # Skip messages with invalid content
            formatted.append({"role": msg["role"], "content": msg["content"]})
        else:
            print(f"[ERROR] Malformed message detected: {msg}")  # ✅ Debugging
    return formatted

async def call_lmstudio_model(messages, objective, model):
    """
    Calls the LMStudio API with OpenAI-style streaming and ensures proper message formatting.
    """

    config = Config()
    api_source, api_value = config.get_api_source()  # Get API source (LMStudio or OpenAI)

    # Ensure LMStudio API URL has the correct OpenAI-compatible endpoint
    if api_source == "lmstudio":
        api_url = api_value.rstrip("/") + "/v1/chat/completions"  # ✅ Ensure correct endpoint
    else:
        raise RuntimeError("No valid API source found.")

    headers = {"Content-Type": "application/json"}

    # ✅ Extract OCR & YOLO data and properly format it as a string
    preprocessed_data_message = None
    if len(messages) > 1 and "Preprocessed data with OCR and YOLO" in messages[1]["content"]:
        preprocessed_data_message = {
            "role": "system",
            "content": format_preprocessed_data(json.loads(messages[1]["content"]))  # ✅ Convert dictionary to string
        }

    # ✅ Ensure valid message format
    formatted_messages = [{"role": "system", "content": objective}] + format_messages(messages)

    # ✅ If preprocessed data exists, overwrite its original entry
    if preprocessed_data_message:
        formatted_messages[1] = preprocessed_data_message  # ✅ Ensures a single message with formatted data

    payload = {
        "model": model,
        "messages": formatted_messages,  # ✅ Use fixed messages format
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,  # ✅ Enable streaming response
    }

    try:
        print(f"[DEBUG] Sending request to {api_source.upper()} API ({api_url}) with model: {model}")
        print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")  # ✅ Pretty print payload for debugging

        with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()

            full_response = ""  # ✅ Store the full response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data: "):  # ✅ Handle OpenAI streaming format
                        try:
                            json_data = json.loads(decoded_line[6:])  # Remove "data: "
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                # ✅ Append content without extra line breaks
                                full_response += content
                                print(content, end="", flush=True)  # ✅ Stream smoothly in console

                                # ✅ Stop if `finish_reason` is set
                                if json_data["choices"][0].get("finish_reason") is not None:
                                    break
                        except json.JSONDecodeError:
                            print("[ERROR] Failed to parse streamed JSON chunk:", decoded_line)

        print("\n[DEBUG] Full Response Received:", full_response)
        return full_response if full_response.strip() else "[ERROR] No valid response from the model."

    except requests.RequestException as e:
        print(f"[ERROR] Failed to connect to {api_source.upper()} API: {e}")
        return f"[ERROR] API connection failed: {e}"

    except Exception as e:
        print(f"[ERROR] Unexpected error in call_lmstudio_model: {e}")
        return f"[ERROR] Unexpected error: {e}"

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

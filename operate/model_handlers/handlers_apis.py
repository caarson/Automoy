from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
from operate.model_handlers.openai_handler import call_openai_model
from operate.model_handlers.lmstudio_handler import call_lmstudio_model
from operate.exceptions import ModelNotRecognizedException
from operate.config import Config  # Import Config class
import json

async def get_next_action(model, messages, objective, session_id, screenshot_path):
    print(f"[handlers_api] Using model: {model}")

    # Perform preprocessing with OCR and YOLO, passing the screenshot path
    # Suppose your preprocess function now returns (summary_string, full_data_dict)
    summary_string, full_data = await preprocess_with_ocr_and_yolo(screenshot_path)
    print(f"[DEBUG] Summary: {summary_string}")
    print(f"[DEBUG] Full Data: {full_data}")

    # Perform preprocessing with OCR and YOLO, passing the screenshot path
    combined_results = await preprocess_with_ocr_and_yolo(screenshot_path)
    print(f"[DEBUG] Preprocessing Results: {combined_results}")

    # Add preprocessing results to messages
    # Ensure preprocessed data is formatted correctly as a string
    if isinstance(combined_results, dict):
        # Convert JSON to a readable string format
        formatted_data = "\n".join(f"{k}: {v}" for k, v in combined_results.items())
    elif isinstance(combined_results, list):
        # Join list elements into a readable format
        formatted_data = "\n".join(combined_results)
    else:
        formatted_data = str(combined_results)  # Ensure it's a string

    # Add the summary to the messages
    messages.append({
        "role": "system",
        "content": summary_string
    })




    config = Config()  # Ensure Config is initialized
    api_source, _ = config.get_api_source()

    if api_source == "openai":
        response = await call_openai_model(messages, objective, model)
        print(f"[DEBUG] OpenAI Response: {response}")  # ✅ Debug Response
        return (response, session_id, full_data)

    if api_source == "lmstudio":
        response = await call_lmstudio_model(messages, objective, model)
        print(f"[DEBUG] LMStudio Response: {response}")  # ✅ Debug Response
        return (response, session_id, full_data)

    raise ModelNotRecognizedException(model)

from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
from operate.model_handlers.openai_handler import call_openai_model
from operate.model_handlers.lmstudio_handler import call_lmstudio_model
from operate.exceptions import ModelNotRecognizedException
from operate.config import Config  # Import Config class

async def get_next_action(model, messages, objective, session_id, screenshot_path):
    print(f"[handlers_api] Using model: {model}")

    # Perform preprocessing with OCR and YOLO, passing the screenshot path
    combined_results = await preprocess_with_ocr_and_yolo(screenshot_path)
    print(f"[DEBUG] Preprocessing Results: {combined_results}")

    # Add preprocessing results to messages
    messages.append({
        "role": "system",
        "content": f"Preprocessed data with OCR and YOLO: {combined_results}"
    })

    config = Config()  # Ensure Config is initialized
    api_source, _ = config.get_api_source()

    if api_source == "openai":
        response = await call_openai_model(messages, objective, model)
        print(f"[DEBUG] OpenAI Response: {response}")  # ✅ Debug Response
        return response, None

    if api_source == "lmstudio":
        response = await call_lmstudio_model(messages, objective, model)
        print(f"[DEBUG] LMStudio Response: {response}")  # ✅ Debug Response
        return response, None

    raise ModelNotRecognizedException(model)

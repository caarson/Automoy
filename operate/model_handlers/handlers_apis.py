from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
from operate.model_handlers.openai_handler import call_openai_model
from operate.model_handlers.lmstudio_handler import call_lmstudio_model

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

    # Route to the appropriate model handler
    if model.startswith("gpt") or "openai" in model:
        if model == "gpt-4-with-ocr-and-yolo":
            model = "gpt-4"
            response = await call_openai_model(messages, objective, model)
            print(f"[DEBUG] OpenAI Response: {response}")
            return response, None  # Ensure this is returning a valid operations list

    if model.startswith("lmstudio"):
        response = await call_lmstudio_model(messages, objective, model)
        print(f"[DEBUG] LMStudio Response: {response}")
        return response, None

    raise ModelNotRecognizedException(f"Model '{model}' not recognized.")


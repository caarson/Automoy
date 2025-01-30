from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
from operate.model_handlers.openai_handler import call_openai_model
from operate.model_handlers.lmstudio_handler import call_lmstudio_model

async def get_next_action(model, messages, objective, session_id):
    print(f"[handlers_api] Using model: {model}")

    # Perform preprocessing with OCR and YOLO
    combined_results = await preprocess_with_ocr_and_yolo()

    # Add preprocessing results to messages
    messages.append({
        "role": "system",
        "content": f"Preprocessed data with OCR and YOLO: {combined_results}"
    })

    # Route to the appropriate model handler
    if model.startswith("gpt") or "openai" in model:
        return await call_openai_model(messages, objective, model), None

    if model.startswith("lmstudio"):
        return await call_lmstudio_model(messages, objective, model), None

    raise ModelNotRecognizedException(f"Model '{model}' not recognized.")

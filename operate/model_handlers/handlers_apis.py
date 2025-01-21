from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
from ollama_handler import call_ollama_model
from openai_handler import call_openai_model
from deepseek_handler import call_deepseek_model

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

    if model.startswith("ollama") or "llama" in model:
        return call_ollama_model(messages, objective, model), None

    if model.startswith("deepseek"):
        return call_deepseek_model(messages, objective, model), None

    raise ModelNotRecognizedException(f"Model '{model}' not recognized.")

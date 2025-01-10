from operate.model_handlers.openai_handler import call_openai_model
from operate.model_handlers.ollama_handler import call_ollama_model
from operate.model_handlers.deepseek_handler import call_deepseek_model
from operate.exceptions import ModelNotRecognizedException

async def get_next_action(model, messages, objective, session_id):
    """
    Main entry point for requesting the next action from a given model.
    Args:
        model (str): The model to use (e.g., "gpt-4", "ollama-llama2", "deepseek").
        messages (list): The list of messages for context.
        objective (str): The task objective.
        session_id (str): A unique session identifier.
    """
    print(f"[handlers_api] Using model: {model}")

    if model.startswith("gpt") or "openai" in model:
        return await call_openai_model(messages, objective, model), None

    if model.startswith("ollama") or "llama" in model:
        return call_ollama_model(messages, objective, model), None

    if model.startswith("deepseek"):
        return call_deepseek_model(messages, objective, model), None

    raise ModelNotRecognizedException(f"Model '{model}' not recognized.")

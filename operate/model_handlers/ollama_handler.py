import ollama

def call_ollama_model(messages, objective, model_name="llama2"):
    """
    Calls the specified Ollama model (LLaMA family).
    Args:
        messages (list): Contextual conversation messages.
        objective (str): Task objective.
        model_name (str): The Ollama model to use.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
        )
        content = response["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"Error calling Ollama model {model_name}: {e}")
        return None

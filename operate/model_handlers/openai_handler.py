from operate.config import Config
import json
import traceback

config = Config()

async def call_openai_model(messages, objective, model_name="gpt-4"):
    """
    Calls OpenAI's GPT-based models.
    Args:
        messages (list): Contextual conversation messages.
        objective (str): Task objective.
        model_name (str): The OpenAI model to use (default is "gpt-4").
    """
    try:
        client = config.initialize_openai()
        response = client.chat.completions.create(
            model=model_name,  # Pass the model name
            messages=messages,
            presence_penalty=1,
            frequency_penalty=1,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        if config.verbose:
            traceback.print_exc()
        return None

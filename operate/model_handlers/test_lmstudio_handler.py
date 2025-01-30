import asyncio
from lmstudio_handler import call_lmstudio_model

async def test_lmstudio():
    test_messages = [{"role": "user", "content": "tell me about google"}]
    test_objective = "Answer the user's question."
    model = "llama-3.2-1b-instruct"  # Your LM Studio model ID

    try:
        response = await call_lmstudio_model(test_messages, test_objective, model)
        print("Test Response:", response)
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    asyncio.run(test_lmstudio())

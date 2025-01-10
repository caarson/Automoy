import ollama

class OllamaHandler:
    def call_ollama_llava(self, messages):
        """
        Calls Ollama LLaVA model and returns the response.
        """
        try:
            response = ollama.chat(model="llava", messages=messages)
            return response["message"]["content"].strip()
        except ollama.ResponseError as e:
            print(f"Ollama Error: {e}")
            return "Ollama service error."

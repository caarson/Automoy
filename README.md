## What is Automoy?
The project is originally derived from Self-Operating Computer by OthersideAI (https://github.com/OthersideAI/self-operating-computer) - Automoy extends on this API and makes deploying an autonomous agent easy. This version uses GPU to accelerate the OCR and object recognition tasks.

Automoy is built for Deepseek, Ollama, and GPT. We will not be supporting Claude and Gemini for the near future while this project remains in development.

## What are the prerequisites?
Automoy only uses CUDA, before use, the application will test the system to see if CUDA is properly configured and the corresponding PyTorch package is installed. If not, the program will not start.


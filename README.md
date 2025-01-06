## What is Automoy?
The project is originally derived from Self-Operating Computer by OthersideAI (https://github.com/OthersideAI/self-operating-computer) - Automoy extends on this API and makes deploying an autonomous agent easy. 

This version uses GPU to accelerate OCR and object recognition tasks.

Automoy is built for Deepseek, Ollama, and GPT. We will not be supporting Claude and Gemini for the near future while this project remains in development.

## What are the prerequisites?

Automoy will require a GPU with atleast 2GB of VRAM - this is in addition to ~ 4GB of free RAM on the system. Caching RAM will make it painfully slow, we recommend that you use a system with atleast 8GB of system RAM, 16GB being ideal.

Automoy only uses CUDA, before use, the application will test the system to see if CUDA is properly configured and the corresponding PyTorch package is installed. If not, the program will not start.

_______________________________________________

There are toolchains that allow for CUDA on AMD GPUs, see below:

SCALE Toolkit:
https://wccftech.com/nvidia-cuda-directly-run-on-amd-gpus-using-scale-toolkit/#:~:text=AnnouncementHardware-,NVIDIA%20CUDA%20Can%20Now%20Directly%20Run,GPUs%20Using%20The%20%E2%80%9CSCALE%E2%80%9D%20Toolkit&text=British%20startup%20Spectral%20Compute%20has,function%20seamlessly%20on%20AMD's%20GPUs.

ZLUDA Toolkit:
https://www.xda-developers.com/nvidia-cuda-amd-zluda/#:~:text=The%20project%20was%20undertaken%20by,and%20Intel%20are%20now%20uninterersted.&text=As%20Phoronix%20reports%2C%20CUDA%2Denabled,also%20work%20on%20Windows%20machines.

We do plan on eventually automating the install process, which includes proper configuration and use of AMD GPUs, but since Automoy is humanly coded, iterations will be slower - please allow us your patience!

For now, you are on your own for getting these prerequisites installed on your system.

_______________________________________________
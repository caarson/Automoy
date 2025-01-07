![Automation Platform Logo](https://cloud.innovatingsustainabletechnology.com/core/preview?fileId=308&x=3840&y=2160&a=true&etag=0af3c5f5d6f560c814fa31e84bc93c62)
## What is Automoy?
The project is originally derived from Self-Operating Computer by OthersideAI (https://github.com/OthersideAI/self-operating-computer) - Automoy extends on this API and makes deploying an autonomous agent easy. 

This version uses GPU to accelerate OCR and object recognition tasks.

Automoy is built for Deepseek, Ollama, and GPT. We will not be supporting Claude and Gemini for the near future while this project remains in development.

## How does Automoy function?
Automoy uses "Computer Vision" meaning that it is analyzing the pictures it is sent, extracting information, and taking sensible actions to complete an objective.

Automoy is build for computer use now, but will later branch out in robotics as the API becomes more refined and as we improve on Computer Vision. Eventually, we imagine real-time analysis of frames, allowing for seamless interaction.

## What are the prerequisites?
Automoy will require a GPU with atleast 2GB of VRAM - this is in addition to ~ 4GB of free RAM on the system. Caching RAM will make it painfully slow, we recommend that you use a system with atleast 8GB of system RAM, 16GB being ideal.

Automoy only uses CUDA, before use, the application will test the system to see if CUDA is properly configured and the corresponding PyTorch package is installed. If not, the program will not start.
We are testing on CUDA 12.4, however, older versions of CUDA may be supported. This remains untested and we hope to get feedback.

Automoy only can be deployed on Windows 10+ machines; though in theory you could run it on any computer with slight modifcations or cosmetic defects. We have not tested this and do not recommend it.
_______________________________________________

There are toolchains that allow for CUDA on AMD GPUs, see below:

SCALE Toolkit:
https://wccftech.com/nvidia-cuda-directly-run-on-amd-gpus-using-scale-toolkit/#:~:text=AnnouncementHardware-,NVIDIA%20CUDA%20Can%20Now%20Directly%20Run,GPUs%20Using%20The%20%E2%80%9CSCALE%E2%80%9D%20Toolkit&text=British%20startup%20Spectral%20Compute%20has,function%20seamlessly%20on%20AMD's%20GPUs.

ZLUDA Toolkit:
https://www.xda-developers.com/nvidia-cuda-amd-zluda/#:~:text=The%20project%20was%20undertaken%20by,and%20Intel%20are%20now%20uninterersted.&text=As%20Phoronix%20reports%2C%20CUDA%2Denabled,also%20work%20on%20Windows%20machines.

We do plan on eventually automating the install process, which includes proper configuration and use of AMD GPUs, but since Automoy is humanly coded, iterations will be slower - please allow us your patience!

For now, you are on your own for getting these prerequisites installed on your system.

_______________________________________________
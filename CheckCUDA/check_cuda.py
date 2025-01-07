import torch
import time

def check_cuda():
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Ensuring CUDA is available
    if torch.cuda.is_available():
        x = torch.rand(5000, 5000).cuda()
        y = torch.rand(5000, 5000).cuda()
        
        start_time = time.time()  # Start timing
        torch.matmul(x, y)        # Perform matrix multiplication
        end_time = time.time()    # End timing
        
        print(f"Matrix multiplication took {end_time - start_time} seconds")
        cuda_enabled = True
    else:
        cuda_enabled = False


    return cuda_enabled
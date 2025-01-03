import torch
import time

# Ensuring CUDA is available
if torch.cuda.is_available():
    x = torch.rand(5000, 5000).cuda()
    y = torch.rand(5000, 5000).cuda()
    
    start_time = time.time()  # Start timing
    torch.matmul(x, y)        # Perform matrix multiplication
    end_time = time.time()    # End timing
    
    print(f"Matrix multiplication took {end_time - start_time} seconds")
else:
    print("CUDA is not available")

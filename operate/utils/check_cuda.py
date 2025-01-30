import torch
import subprocess
import os
import re
from ultralytics import YOLO

def run_yolo_checks():
    """
    Runs the 'yolo checks' command to validate YOLO's environment setup and checks if a CUDA version is listed.
    """
    try:
        print("Running 'yolo checks' for additional diagnostics...")
        result = subprocess.run(["yolo", "checks"], capture_output=True, text=True)
        output = result.stdout
        print(output)

        # Regex pattern to detect a CUDA version (e.g., "CUDA 12.4", "CUDA 11.8", etc.)
        cuda_version_pattern = r"CUDA\s+\d+\.\d+"

        # If a CUDA version is found, CUDA is active
        if re.search(cuda_version_pattern, output):
            cuda_version = re.search(cuda_version_pattern, output).group(0)
            print(f"YOLO environment checks confirm that CUDA is available: {cuda_version}")
            return True
        else:
            print("YOLO environment checks indicate that no CUDA version was listed. CUDA may not be available.")
            return False

    except FileNotFoundError:
        print("The 'yolo' CLI is not installed. Please install YOLO CLI with 'pip install ultralytics'.")
        return False

def get_test_image_path():
    """
    Returns the path to the test image in the expected directory. Ensures compatibility with the Anaconda environment.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Navigate out of utils to project/
    test_image_path = os.path.abspath(os.path.join(base_dir, "data", "YOLO", "test_image", "image.jpeg"))

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}. Please ensure it is included in the setup.")

    return test_image_path

def check_cuda(min_memory_gb=2):
    """
    Checks if CUDA is available, the GPU has enough memory, and YOLO detects the GPU.
    Args:
        min_memory_gb (int): Minimum required GPU memory in GB.
    Returns:
        bool: True if CUDA and YOLO both detect the GPU, and it meets the minimum memory requirement.
    """
    print("Checking CUDA and YOLO GPU availability...")

    # Check PyTorch CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU setup.")
        return False

    # Run YOLO environment checks
    if not run_yolo_checks():
        print("YOLO environment checks failed. Please ensure CUDA is properly configured.")
        return False

    # Get CUDA device info
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert bytes to GB
    print(f"Using device: {device_name}")
    print(f"Total GPU memory: {total_memory:.2f} GB")

    if total_memory < min_memory_gb:
        print(f"Insufficient GPU memory. Required: {min_memory_gb} GB, Available: {total_memory:.2f} GB")
        return False

    try:
        print("Testing YOLO GPU support...")

        # Load YOLO model and force it to run on GPU
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Navigate out of utils to project/
        model_path = os.path.abspath(os.path.join(base_dir, "data", "YOLO", "test_models", "yolov8n.pt"))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}. Please ensure it is included in the setup.")
        
        yolo_model = YOLO(model_path).to("cuda")
        test_image_path = get_test_image_path()

        print(f"Using test image: {test_image_path}")
        print("Performing YOLO inference on GPU...")

        # Perform YOLO inference on the test image
        prediction = yolo_model.predict(source=test_image_path, imgsz=640, conf=0.25)
        actual_device = str(yolo_model.device)

        if "cuda" in actual_device:
            print(f"YOLO successfully utilized GPU: {device_name} (Device: {actual_device})")
            print(f"Inference complete: {len(prediction)} objects detected.")
            return True
        else:
            print(f"Warning: YOLO reports it ran on {actual_device} instead of GPU.")
            return False

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        return False

    except Exception as e:
        print(f"YOLO GPU test failed: {e}")
        return False

# Main test
if __name__ == "__main__":
    print("Starting CUDA and YOLO GPU Test...")
    if check_cuda():
        print("CUDA, PyTorch and YOLO are successfully configured and GPU is available.")
    else:
        print("CUDA and/or YOLO and/or PyTorch GPU configuration failed. Please check your setup.")

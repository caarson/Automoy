import subprocess
import math
import os
from threading import Timer
import json
from datetime import datetime
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F

# --- GPU Memory Management Imports ---
import wmi
import ctypes
import pynvml

# --- docTR imports for DETECTION ---
from doctr.io import DocumentFile
from doctr.models import detection

# --- TrOCR imports for RECOGNITION ---
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- Local config import, if you have one ---
from operate.config import Config

# Initialize pynvml and set availability flag
try:
    pynvml.nvmlInit()
    pynvml_available = True
    pynvml.nvmlShutdown()
except (ImportError, pynvml.NVMLError):
    pynvml_available = False

# Load configuration
config = Config()

# Dynamic path setup
BASE_DIR = os.path.dirname(__file__)
OCR_DIR = os.path.join(BASE_DIR, "..", "ocr")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "..", "screenshots", "screenshot.png")

# Ensure OCR directory exists
os.makedirs(OCR_DIR, exist_ok=True)

# Configure CUDA shared memory usage without 'expandable_segments'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# Timeout class declaration
class TimeoutException(Exception):
    pass

# -----------------------------
# GPU Memory Management Functions
# -----------------------------

def get_gpu_memory_task_manager():
    """
    Retrieve GPU memory statistics for the NVIDIA GPU using WMI and dynamic shared memory query.
    """
    try:
        # Connect to WMI
        w = wmi.WMI(namespace="root\\CIMV2")
        video_controllers = w.query("SELECT AdapterRAM, Name, DriverVersion FROM Win32_VideoController")

        for controller in video_controllers:
            adapter_name = controller.Name

            # Focus only on NVIDIA GPUs
            if "NVIDIA" in adapter_name:
                adapter_ram = controller.AdapterRAM
                driver_version = controller.DriverVersion

                # Validate and calculate dedicated memory
                if adapter_ram is not None and adapter_ram > 0:
                    dedicated_memory_gb = adapter_ram / (1024 ** 3)
                else:
                    # Fallback to nvidia-smi if WMI fails to retrieve AdapterRAM
                    dedicated_memory_gb = get_dedicated_memory_from_nvidia_smi()

                # Retrieve free shared memory with proper validation
                shared_memory_gb = get_free_shared_memory()

                # Print GPU details
                print(f"Adapter: {adapter_name}")
                print(f"Driver Version: {driver_version}")
                print(f"Dedicated Memory: {dedicated_memory_gb:.2f} GB")
                print(f"Shared Memory: {shared_memory_gb:.2f} GB")

                # Return results
                return dedicated_memory_gb, shared_memory_gb

        # If no NVIDIA GPU is found, return 0
        print("No NVIDIA GPU found.")
        return 0, 0

    except Exception as e:
        print(f"[ERROR] Error retrieving GPU memory: {e}")
        return 0, 0

def get_free_shared_memory(fallback_shared_gb=6.0, max_shared_gb=15.9, tolerance_gb=0.5):
    """
    Retrieve free shared memory dynamically and handle fluctuations.

    :param fallback_shared_gb: Fallback shared memory in GB if the actual value is invalid (default: 6.0 GB).
    :param max_shared_gb: Maximum possible shared memory in GB (default: 15.9 GB).
    :param tolerance_gb: Tolerance to handle small fluctuations (default: 0.5 GB).
    :return: Free shared memory in GB.
    """
    try:
        # Connect to WMI
        w = wmi.WMI(namespace="root\\CIMV2")
        os_info = w.query("SELECT FreePhysicalMemory FROM Win32_OperatingSystem")

        if os_info and len(os_info) > 0:
            free_physical_memory_kb = int(os_info[0].FreePhysicalMemory)
            free_memory_gb = free_physical_memory_kb / (1024 ** 2)  # Convert KB to GB

            # Validate the free memory within expected ranges
            if free_memory_gb > max_shared_gb:
                print(f"[WARNING] Shared memory reported unusually high: {free_memory_gb:.2f} GB. Using max limit.")
                return max_shared_gb
            elif free_memory_gb < fallback_shared_gb - tolerance_gb:
                print(f"[WARNING] Shared memory reported unusually low: {free_memory_gb:.2f} GB. Using fallback value.")
                return fallback_shared_gb
            return free_memory_gb

        print("[WARNING] Unable to retrieve free shared memory. Using fallback value.")
        return fallback_shared_gb

    except Exception as e:
        print(f"[ERROR] Error retrieving free shared memory: {e}. Using fallback value.")
        return fallback_shared_gb

def get_dedicated_memory_from_nvidia_smi():
    """
    Retrieve dedicated GPU memory using nvidia-smi as a fallback.
    """
    try:
        # Run nvidia-smi and parse the output
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        dedicated_memory_mb = int(output.strip())  # Value is in MB
        dedicated_memory_gb = dedicated_memory_mb / 1024  # Convert to GB
        return dedicated_memory_gb
    except Exception as e:
        print(f"Error retrieving dedicated memory from nvidia-smi: {e}")
        return 0

def get_total_gpu_memory_bytes():
    """
    Calculates the total GPU memory in bytes by summing dedicated and shared memory.
    """
    dedicated_memory_gb, shared_memory_gb = get_gpu_memory_task_manager()
    total_memory_gb = dedicated_memory_gb + shared_memory_gb
    total_memory_bytes = total_memory_gb * (1024 ** 3)
    print(f"Total GPU Memory Limit: {total_memory_gb:.2f} GB")
    return total_memory_bytes

def get_gpu_memory_info():
    """
    Retrieves and prints detailed GPU memory information.
    """
    dedicated_memory_gb, shared_memory_gb = get_gpu_memory_task_manager()
    total_memory_gb = dedicated_memory_gb + shared_memory_gb
    print(f"Total GPU Memory Limit: {total_memory_gb:.2f} GB")

def get_gpu_memory_usage():
    """
    Retrieve the current GPU memory usage for both dedicated and shared memory.
    
    :return: A tuple (used_memory_gb, free_memory_gb, total_memory_gb)
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU setup
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Dedicated memory usage
        used_dedicated_memory_gb = mem_info.used / (1024 ** 3)
        free_dedicated_memory_gb = mem_info.free / (1024 ** 3)
        total_dedicated_memory_gb = mem_info.total / (1024 ** 3)

        # Retrieve free shared memory with proper validation
        free_shared_memory_gb = get_free_shared_memory()
        total_shared_memory_gb = 15.9  # Maximum shared memory assumption

        # Ensure shared memory reports are logical
        if free_shared_memory_gb > total_shared_memory_gb:
            free_shared_memory_gb = total_shared_memory_gb
        if free_shared_memory_gb < 0:
            free_shared_memory_gb = 0

        # Combine dedicated and shared memory for total calculations
        used_shared_memory_gb = total_shared_memory_gb - free_shared_memory_gb
        used_total_memory_gb = used_dedicated_memory_gb + used_shared_memory_gb
        free_total_memory_gb = free_dedicated_memory_gb + free_shared_memory_gb
        total_memory_gb = total_dedicated_memory_gb + total_shared_memory_gb

        pynvml.nvmlShutdown()

        return used_total_memory_gb, free_total_memory_gb, total_memory_gb
    except pynvml.NVMLError as e:
        print(f"[ERROR] Unable to retrieve GPU memory usage: {e}")
        return 0, 0, 0

# -----------------------------
# Device Initialization (MUST COME FIRST)
# -----------------------------
device = torch.device("cuda")
print(f"[INFO] Using device: {device}")

# -----------------------------
# Initialize DETECTION (docTR) - FIXED PRECISION
# -----------------------------
try:
    # Load detection model with full precision
    det_model = detection.db_resnet50(pretrained=True)
    det_model = det_model.eval().to(device)
    print("[SUCCESS] Detection model loaded in FP32")
except Exception as e:
    print(f"[ERROR] Failed to initialize detection model: {e}")
    raise

# -----------------------------
# Initialize RECOGNITION (TrOCR)
# -----------------------------
try:
    # Load processor with fast tokenizer
    trocr_name = "microsoft/trocr-base-stage1"
    processor = TrOCRProcessor.from_pretrained(trocr_name, use_fast=True)
    
    # Load model with mixed precision
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        trocr_name,
        torch_dtype=torch.float16
    ).eval().to(device)
    
    # Warmup GPU with dummy inference using correct input size
    with torch.no_grad(), torch.cuda.amp.autocast():
        dummy_input = torch.randn(1, 3, 384, 384, device=device, dtype=torch.float16)  # Changed to 384x384
        _ = trocr_model.generate(dummy_input, max_length=10)
    
    print("[SUCCESS] Recognition model loaded and warmed up on GPU")

except Exception as e:
    print(f"[ERROR] Failed to initialize recognition model: {e}")
    raise

# Ensure models are in eval mode
det_model.eval()
trocr_model.eval()

# -----------------------------
# Memory Management Functions
# -----------------------------

def estimate_detection_memory(h, w, channels=3, dtype_bytes=4, overhead_factor=4.0):
    """
    Estimate total GPU memory usage (in bytes) for detection on a single image
    of shape (C=channels, H, W). Multiply raw input size by an overhead
    factor to account for intermediate feature maps, etc.

    :param h: image height
    :param w: image width
    :param channels: typically 3 (RGB)
    :param dtype_bytes: typically 4 bytes for float32
    :param overhead_factor: multiplier for overhead
    :return: estimated GPU memory usage in bytes
    """
    input_bytes = h * w * channels * dtype_bytes
    total_estimated = input_bytes * overhead_factor
    return total_estimated

def compute_scale_factor(page_tensor, memory_limit, overhead_factor=4.0):
    """
    Compute a scale factor based on the memory limit and the image size.

    :param page_tensor: CHW float tensor
    :param memory_limit: available memory in bytes
    :param overhead_factor: multiplier for overhead
    :return: scale factor <=1.0
    """
    c, h, w = page_tensor.shape
    estimated_mem = estimate_detection_memory(h, w, channels=c, overhead_factor=overhead_factor)
    if estimated_mem <= memory_limit:
        return 1.0
    # Calculate scale factor based on sqrt of memory ratio
    scale_factor = math.sqrt(memory_limit / estimated_mem)
    scale_factor = min(scale_factor, 1.0)  # Ensure not to upscale
    return scale_factor

def detect_page_with_dynamic_scaling(page_tensor, model, memory_limit, min_scale=0.25, scale_step=0.9, overhead_margin=0.9):
    """
    Attempt to run docTR detection on a single page tensor with dynamic scaling.
    Starts with a computed scale factor and reduces it iteratively until it fits within memory constraints.

    :param page_tensor: CHW float tensor on CPU
    :param model: docTR detection model (on GPU)
    :param memory_limit: available memory in bytes
    :param min_scale: minimum scale factor to prevent excessive downscaling
    :param scale_step: factor by which to reduce scale each step (default: 0.9 for gradual adjustment)
    :param overhead_margin: multiplier to reserve memory buffer (default: 0.9 to use 90% of available memory)
    :return: detection_results (dictionary), scale_factor actually used
    """
    # Apply margin to memory limit
    adjusted_memory_limit = memory_limit * overhead_margin
    print(f"[INFO] Adjusted memory limit with overhead margin: {adjusted_memory_limit / (1024 ** 3):.2f} GB")

    # Initial scale factor based on memory estimation
    scale_factor = compute_scale_factor(page_tensor, adjusted_memory_limit)
    if scale_factor < min_scale:
        scale_factor = min_scale

    while scale_factor >= min_scale:
        try:
            # Resize image if scale_factor < 1.0
            if scale_factor < 1.0:
                new_h = int(page_tensor.shape[1] * scale_factor)
                new_w = int(page_tensor.shape[2] * scale_factor)
                if new_h < 1 or new_w < 1:
                    raise RuntimeError("Scale factor too small, image dimensions reduced below 1.")
                scaled_page = F.resize(page_tensor, (new_h, new_w))
            else:
                scaled_page = page_tensor

            # Pad to make dimensions divisible by 32
            h, w = scaled_page.shape[1], scaled_page.shape[2]
            padded_h = (h + 31) // 32 * 32
            padded_w = (w + 31) // 32 * 32
            padding = [0, padded_w - w, 0, padded_h - h]  # left, right, top, bottom
            scaled_page_padded = F.pad(scaled_page, padding, fill=0)

            # Prepare batch
            input_batch = scaled_page_padded.unsqueeze(0).to(device)

            # Attempt detection
            with torch.no_grad():
                detection_results = model(input_batch)

            print(f"[INFO] Detection succeeded at scale_factor={scale_factor:.3f}")
            return detection_results, scale_factor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM at scale_factor={scale_factor:.3f}. Reducing scale.")
                torch.cuda.empty_cache()
                # Gradually reduce scale_factor for better resolution
                scale_factor *= scale_step
            else:
                raise e

    raise RuntimeError("Cannot process image within memory constraints even at minimum scale.")

# In the adaptive_chunking function definition, add max_chunk_size parameter
def adaptive_chunking(tensors, model_or_generate_fn, memory_limit, 
                      initial_chunk_size=8,  # Reduced from 16
                      max_chunk_size=32,     # Reduced from 64
                      min_chunk_size=1):
    """
    Dynamically process tensors in chunks while maximizing GPU memory utilization.
    Added max_chunk_size to prevent oversized batches during generation.
    """
    results = []
    i = 0
    total = tensors.size(0)
    chunk_size = initial_chunk_size
    print(f"[INFO] Total number of tensors: {total}")

    while i < total:
        # Enforce maximum chunk size
        current_chunk_size = min(chunk_size, total - i, max_chunk_size)
        chunk = tensors[i:i + current_chunk_size].to(device)
        print(f"[DEBUG] Processing chunk {i // current_chunk_size + 1} with size {current_chunk_size}...")

        try:
            # Conservative memory estimation with generation overhead factor
            element_size = chunk.element_size()
            estimated_mem_per_element = chunk[0].nelement() * element_size * 10  # 10x conservative multiplier
            safe_batch_size = int((memory_limit * 0.8) // estimated_mem_per_element)
            current_chunk_size = min(current_chunk_size, safe_batch_size, max_chunk_size)
            
            if current_chunk_size < 1:
                raise RuntimeError("Memory requirements exceed available GPU memory")

            chunk = tensors[i:i + current_chunk_size].to(device)

            # Perform GPU operations with optimized generation parameters
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                output = model_or_generate_fn(chunk)
                torch.cuda.synchronize()

            # Collect results and adjust chunk size
            results.append(output)
            i += current_chunk_size
            
            # Dynamic chunk size adjustment
            if chunk_size < max_chunk_size:
                chunk_size = min(chunk_size * 2, max_chunk_size)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM encountered. Reducing chunk size...")
                torch.cuda.empty_cache()
                chunk_size = max(min_chunk_size, chunk_size // 2)
            else:
                raise e

    return torch.cat(results, dim=0) if isinstance(results[0], torch.Tensor) else results

def timeout_handler():
    raise TimeoutException("Recognition process exceeded time limit.")

def timeout_handler():
    raise TimeoutException("Recognition process exceeded time limit.")

def perform_ocr(image_path, min_scale=0.25, scale_step=0.75):
    """
    Perform OCR on the image file while ensuring GPU utilization for all operations.

    :param image_path: Path to the image file.
    :param min_scale: Minimum scaling factor for detection.
    :param scale_step: Reduction factor for scaling in case of OOM.
    :return: List of OCR results.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image as docTR DocumentFile object
    pages = DocumentFile.from_images(image_path)

    final_ocr_results = []

    with Image.open(image_path) as img:
        full_img = img.copy()  # Ensure the image is loaded into memory
        full_width, full_height = full_img.size

    # GPU memory configuration
    memory_limit = get_total_gpu_memory_bytes()
    print(f"[INFO] Total GPU Memory Limit: {memory_limit / (1024 ** 3):.2f} GB")

    for page_idx, page in enumerate(pages):
        print(f"[INFO] Processing page {page_idx + 1}/{len(pages)}")

        # Convert to tensor and normalize properly
        page_tensor = torch.tensor(page).permute(2, 0, 1).float().to(device)
        page_tensor = page_tensor / 255.0  # Keep in float32

        # Dynamic scaling for detection
        try:
            detection_dict, scale_used = detect_page_with_dynamic_scaling(
                page_tensor, det_model, memory_limit, min_scale=min_scale, scale_step=scale_step
            )
        except RuntimeError as e:
            print(f"[ERROR] Failed to detect text on page {page_idx + 1}: {e}")
            continue

        # ADD DEBUG LINES HERE ------------------
        print(f"Raw detection output: {detection_dict}")  # Debug output
        preds = detection_dict["preds"][0] if len(detection_dict["preds"]) > 0 else None
        # ---------------------------------------
        
        if not preds or len(preds["words"]) == 0:
            print(f"[INFO] No text detected on page {page_idx + 1}")
            continue

        cropped_images = []
        polygons_and_conf = []

        # Crop and resize images for recognition
        for box_idx, (xmin, ymin, xmax, ymax, conf) in enumerate(preds["words"]):
            if conf < 0.3:
                continue

            abs_xmin, abs_ymin = xmin * full_width, ymin * full_height
            abs_xmax, abs_ymax = xmax * full_width, ymax * full_height

            crop = full_img.crop((abs_xmin, abs_ymin, abs_xmax, abs_ymax))
            # In perform_ocr() function, change crop size:
            crop_resized = crop.resize((384, 384), Image.Resampling.LANCZOS)  # Was 256x256
            cropped_images.append(crop_resized)

            polygon = [
                [abs_xmin, abs_ymin],
                [abs_xmax, abs_ymin],
                [abs_xmax, abs_ymax],
                [abs_xmin, abs_ymax],
            ]
            polygons_and_conf.append((polygon, conf))

        print(f"[DEBUG] {len(cropped_images)} cropped images prepared for recognition.")

        if len(cropped_images) > 0:
            crop_tensors = processor(images=cropped_images, return_tensors="pt", padding=True).pixel_values.to(device)

            def generate_fn(batch):
                return trocr_model.generate(
                    batch,
                    max_length=40,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                ).sequences

            try:
                texts = []
                print("[INFO] Starting text recognition in batches...")
                generated_ids = adaptive_chunking(crop_tensors, generate_fn, memory_limit)
                if generated_ids is not None:
                    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    print(f"[INFO] Recognition completed for {len(texts)} text segments.")
                else:
                    print(f"[WARNING] No recognition results for page {page_idx + 1}")

            except RuntimeError as e:
                print(f"[ERROR] Recognition failed on page {page_idx + 1}: {e}")
                continue

            # Compile OCR results
            for (polygon, conf), txt in zip(polygons_and_conf, texts):
                final_ocr_results.append([polygon, [txt, conf]])
        else:
            print("[INFO] No valid cropped images for recognition.")

    return final_ocr_results

def store_ocr_results(ocr_data, output_path):
    """
    Store the entire OCR output as JSON in the specified file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=4)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(SCREENSHOT_PATH):
        print(f"Screenshot not found at {SCREENSHOT_PATH}. Please provide a valid image.")
    else:
        try:
            print("Retrieving GPU Memory Information...")
            get_gpu_memory_info()

            print(f"\nPerforming OCR on: {SCREENSHOT_PATH}")
            ocr_results = perform_ocr(SCREENSHOT_PATH)

            if ocr_results:
                # Save raw OCR data
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                ocr_data_path = os.path.join(OCR_DIR, f"ocr_data_{datetime_str}.json")
                store_ocr_results(ocr_results, ocr_data_path)
                print(f"OCR data saved at: {ocr_data_path}")
            else:
                print("No OCR results to save.")

        except Exception as e:
            print(f"Error: {e}")
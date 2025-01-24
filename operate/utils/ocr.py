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
    Retrieve the current GPU memory usage for dedicated memory only.
    
    :return: A tuple (used_memory_gb, free_memory_gb, total_memory_gb)
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU setup
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        used = mem_info.used / (1024 ** 3)
        free = mem_info.free / (1024 ** 3)
        total = mem_info.total / (1024 ** 3)

        pynvml.nvmlShutdown()

        return used, free, total
    except pynvml.NVMLError as e:
        print(f"[ERROR] Unable to retrieve GPU memory usage: {e}")
        return 0, 0, 0

# -----------------------------
# Device Initialization (MUST COME FIRST)
# -----------------------------
device = torch.device("cuda")

# Configure CUDA memory settings
torch.cuda.set_per_process_memory_fraction(0.99)
torch.backends.cudnn.benchmark = True
print(f"[INFO] Using device: {device}")

# -----------------------------
# Initialize DETECTION (docTR) with Mixed Precision
# -----------------------------
try:
    det_model = detection.db_resnet50(pretrained=True).half().eval().to(device)
    print("[SUCCESS] Detection model loaded in FP16")
except Exception as e:
    print(f"[ERROR] Failed to initialize detection model: {e}")
    raise

# -----------------------------
# Initialize RECOGNITION (TrOCR) with Optimized Settings
# -----------------------------
try:
    trocr_name = "microsoft/trocr-base-stage1"
    processor = TrOCRProcessor.from_pretrained(trocr_name, use_fast=True)
    
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        trocr_name,
        torch_dtype=torch.float16
    ).eval().to(device)
    
    # Warmup with updated autocast syntax
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        dummy_input = torch.randn(1, 3, 384, 384, device=device, dtype=torch.float16)
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
    Estimate total GPU memory usage (in bytes) for detection on a single image.
    """
    input_bytes = h * w * channels * dtype_bytes
    return input_bytes * overhead_factor

def compute_scale_factor(page_tensor, memory_limit, overhead_factor=4.0):
    """
    Compute a scale factor based on the memory limit and the image size.
    """
    c, h, w = page_tensor.shape
    estimated_mem = estimate_detection_memory(h, w, channels=c, overhead_factor=overhead_factor)
    if estimated_mem <= memory_limit:
        return 1.0
    scale_factor = math.sqrt(memory_limit / estimated_mem)
    return min(scale_factor, 1.0)

def detect_page_with_dynamic_scaling(page_tensor, model, memory_limit, min_scale=0.25, scale_step=0.9, overhead_margin=0.9):
    """
    Attempt to run docTR detection with dynamic scaling and mixed precision.
    """
    adjusted_memory_limit = memory_limit * overhead_margin
    scale_factor = compute_scale_factor(page_tensor, adjusted_memory_limit)
    scale_factor = max(scale_factor, min_scale)

    while scale_factor >= min_scale:
        try:
            if scale_factor < 1.0:
                new_h = int(page_tensor.shape[1] * scale_factor)
                new_w = int(page_tensor.shape[2] * scale_factor)
                scaled_page = F.resize(page_tensor, (new_h, new_w))
            else:
                scaled_page = page_tensor

            h, w = scaled_page.shape[1], scaled_page.shape[2]
            padded_h = (h + 31) // 32 * 32
            padded_w = (w + 31) // 32 * 32
            padding = [0, padded_w - w, 0, padded_h - h]
            scaled_page_padded = F.pad(scaled_page, padding, fill=0)

            input_batch = scaled_page_padded.unsqueeze(0).to(device, non_blocking=True).half()

            with torch.no_grad(), torch.cuda.amp.autocast():
                detection_results = model(input_batch)

            print(f"[INFO] Detection succeeded at scale_factor={scale_factor:.3f}")
            return detection_results, scale_factor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM at scale_factor={scale_factor:.3f}. Reducing scale.")
                torch.cuda.empty_cache()
                scale_factor *= scale_step
            else:
                raise e

    raise RuntimeError("Cannot process image within memory constraints even at minimum scale.")

def adaptive_chunking(tensors, model_or_generate_fn, memory_limit,
                      initial_chunk_size=256,
                      max_chunk_size=1024,
                      min_chunk_size=128,
                      scaling_factor=1.5):
    """Optimized chunking with CPU-side memory pinning"""
    results = []
    i = 0
    total = tensors.size(0)
    chunk_size = initial_chunk_size
    print(f"[INFO] Total number of tensors: {total}")

    element_size = tensors.element_size()
    elements_per_item = tensors[0].numel()
    
    while i < total:
        try:
            # Get CPU-side chunk and pin memory before transfer
            cpu_chunk = tensors[i:i + current_chunk_size].cpu()
            pinned_chunk = cpu_chunk.pin_memory()
            
            # Async transfer to GPU
            chunk = pinned_chunk.to(device, non_blocking=True)
            
            # Memory usage tracking
            print(f"Memory Usage: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
                  f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model_or_generate_fn(chunk)
                torch.cuda.synchronize()

            results.append(output)
            i += current_chunk_size
            
            # Cleanup with forced GC
            del cpu_chunk, pinned_chunk, chunk, output
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM at chunk {current_chunk_size}. Scaling back...")
                chunk_size = max(min_chunk_size, chunk_size // 2)
                torch.cuda.empty_cache()
            else:
                raise e

    return torch.cat(results, dim=0) if isinstance(results[0], torch.Tensor) else results

def perform_ocr(image_path, min_scale=0.25, scale_step=0.75):
    """OCR pipeline with Windows shared memory support"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    pages = DocumentFile.from_images(image_path)
    final_ocr_results = []

    with Image.open(image_path) as img:
        full_img = img.copy()
        full_width, full_height = full_img.size

    memory_limit = get_total_gpu_memory_bytes()
    print(f"[INFO] Total Available Memory (Dedicated + Shared): {memory_limit / (1024 ** 3):.2f} GB")

    for page_idx, page in enumerate(pages):
        print(f"[INFO] Processing page {page_idx + 1}/{len(pages)}")

        # Initialize processing variables
        page_tensor = torch.tensor(page).permute(2, 0, 1).float().cpu()
        page_tensor = page_tensor / 255.0
        cropped_images = []  # Initialize here to prevent NameError
        polygons_and_conf = []

        try:
            # Detection with shared memory awareness
            detection_dict, _ = detect_page_with_dynamic_scaling(
                page_tensor, det_model, memory_limit,
                min_scale=min_scale,
                scale_step=scale_step
            )
        except RuntimeError as e:
            print(f"[ERROR] Detection failed: {e}")
            continue

        # Process detected boxes
        if detection_dict["preds"]:
            preds = detection_dict["preds"][0]
            for box_idx, (xmin, ymin, xmax, ymax, conf) in enumerate(preds["words"]):
                if conf < 0.3:
                    continue

                # Convert relative coordinates to absolute
                abs_coords = (
                    xmin * full_width,
                    ymin * full_height,
                    xmax * full_width,
                    ymax * full_height
                )
                
                # Crop and resize using Windows-friendly operations
                crop = full_img.crop(abs_coords)
                crop_resized = crop.resize((384, 384), Image.Resampling.LANCZOS)
                cropped_images.append(crop_resized)

                # Store polygon coordinates with confidence
                polygons_and_conf.append((
                    [
                        [abs_coords[0], abs_coords[1]],
                        [abs_coords[2], abs_coords[1]],
                        [abs_coords[2], abs_coords[3]],
                        [abs_coords[0], abs_coords[3]]
                    ],
                    conf
                ))

        if len(cropped_images) > 0:
            # Use Windows-optimized memory pinning
            with torch.cuda.amp.autocast(dtype=torch.float16):  # Updated autocast syntax
                crop_tensors = processor(
                    images=cropped_images, 
                    return_tensors="pt", 
                    padding=True
                ).pixel_values
                
                # Pin memory for shared GPU memory utilization
                pinned_tensors = crop_tensors.pin_memory().to(device, non_blocking=True)

                def generate_fn(batch):
                    return trocr_model.generate(
                        batch,
                        max_length=16,
                        num_beams=1,
                        early_stopping=True,
                        output_scores=True,
                        return_dict_in_generate=True
                    ).sequences

                try:
                    # Process with shared memory chunking
                    generated_ids = adaptive_chunking(
                        pinned_tensors, 
                        generate_fn, 
                        memory_limit,
                        initial_chunk_size=128,
                        max_chunk_size=512
                    )
                    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    # Compile results
                    for (polygon, conf), txt in zip(polygons_and_conf, texts):
                        final_ocr_results.append([polygon, [txt, conf]])

                except RuntimeError as e:
                    print(f"[ERROR] Recognition failed: {str(e)}")
                    gc.collect()
                    torch.cuda.empty_cache()

        # Force cleanup for Windows memory management
        del page_tensor
        gc.collect()
        torch.cuda.empty_cache()

    return final_ocr_results

def store_ocr_results(ocr_data, output_path):
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
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                ocr_data_path = os.path.join(OCR_DIR, f"ocr_data_{datetime_str}.json")
                store_ocr_results(ocr_results, ocr_data_path)
                print(f"OCR data saved at: {ocr_data_path}")
            else:
                print("No OCR results to save.")

        except Exception as e:
            print(f"Error: {e}")
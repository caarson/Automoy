import subprocess
import math
import os
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

                # Retrieve free shared memory dynamically
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
        print(f"Error retrieving GPU memory: {e}")
        return 0, 0

def get_free_shared_memory():
    """
    Retrieve free shared memory dynamically using WMI.
    """
    try:
        # Connect to WMI
        w = wmi.WMI(namespace="root\\CIMV2")
        os_info = w.query("SELECT FreePhysicalMemory FROM Win32_OperatingSystem")

        if os_info and len(os_info) > 0:
            free_physical_memory_kb = int(os_info[0].FreePhysicalMemory)
            free_memory_gb = free_physical_memory_kb / (1024 ** 2)  # Convert KB to GB
            return free_memory_gb

        print("[WARNING] Unable to retrieve free shared memory.")
        return 0

    except Exception as e:
        print(f"Error retrieving free shared memory: {e}")
        return 0


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

# -----------------------------
# Initialize DETECTION (docTR)
# -----------------------------
det_model = detection.db_resnet50(pretrained=True).eval()

# -----------------------------
# Initialize RECOGNITION (TrOCR)
# -----------------------------
trocr_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(trocr_name)
trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_name).eval()

# Move models to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_model = det_model.to(device)
trocr_model = trocr_model.to(device)

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

def detect_page_with_dynamic_scaling(page_tensor, model, memory_limit, min_scale=0.25, scale_step=0.75):
    """
    Attempt to run docTR detection on a single page tensor with dynamic scaling.
    Starts with a computed scale factor and reduces it by scale_step until it fits.

    :param page_tensor: CHW float tensor on CPU
    :param model: docTR detection model (on GPU)
    :param memory_limit: available memory in bytes
    :param min_scale: minimum scale factor to prevent excessive downscaling
    :param scale_step: factor by which to reduce scale each step
    :return: detection_results (dictionary), scale_factor actually used
    """
    # Initial scale factor based on memory estimation
    scale_factor = compute_scale_factor(page_tensor, memory_limit)
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
                # Reduce scale_factor
                scale_factor *= scale_step
            else:
                raise e

    raise RuntimeError("Cannot process image within memory constraints even at minimum scale.")

def adaptive_chunking(tensors, model_or_generate_fn, memory_limit, chunk_size=8):
    """
    Process tensors dynamically in chunks, using available memory but avoiding OOM errors.

    :param tensors: a batch tensor on the correct device
    :param model_or_generate_fn: either a model.forward or model.generate function
    :param memory_limit: available memory in bytes
    :param chunk_size: initial chunk size
    :return: concatenated results
    """
    results = []
    i = 0
    total = tensors.size(0)
    while i < total:
        current_chunk_size = min(chunk_size, total - i)
        chunk = tensors[i:i + current_chunk_size]
        try:
            # Rough memory check: ensure chunk doesn't exceed 50% of memory_limit
            # to provide buffer for model operations
            memory_required = chunk.element_size() * chunk.nelement()
            if memory_required > (memory_limit * 0.5):
                # Reduce chunk size proportionally
                scaling = (memory_limit * 0.5) / memory_required
                adjusted_chunk_size = max(1, int(current_chunk_size * scaling))
                if adjusted_chunk_size < current_chunk_size:
                    print(f"[INFO] Adjusting chunk size from {current_chunk_size} to {adjusted_chunk_size}")
                    chunk = chunk[:adjusted_chunk_size]
                    current_chunk_size = adjusted_chunk_size

            # Run the model or generate function
            output = model_or_generate_fn(chunk)

            # Collect results
            if isinstance(output, torch.Tensor):
                results.append(output)
            elif isinstance(output, list):
                # Assume list of tensors (e.g., from generate)
                results.extend(output)
            else:
                raise TypeError("Model output type not supported in adaptive_chunking.")

            i += current_chunk_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] Chunk starting at index {i} caused OOM. Reducing chunk size.")
                torch.cuda.empty_cache()
                # Reduce chunk_size exponentially
                chunk_size = max(1, chunk_size // 2)
            else:
                raise e

    if not results:
        return None

    if isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=0)
    else:
        # For list outputs, such as from generate()
        return results

def perform_ocr(image_path, min_scale=0.25, scale_step=0.75):
    """
    End-to-end OCR using docTR for detection and TrOCR for recognition.
    Dynamically adjusts processing to avoid exceeding memory limits.

    :param image_path: Path to the image file
    :param min_scale: Minimum scaling factor
    :param scale_step: Factor by which to reduce scale when OOM occurs
    :return: List of OCR results
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the image as a docTR DocumentFile object (list of pages as NumPy arrays)
    pages = DocumentFile.from_images(image_path)  # Returns list of np.array([...])

    final_ocr_results = []

    with Image.open(image_path) as full_img:
        full_width, full_height = full_img.size

    # Get total GPU memory limit
    memory_limit = get_total_gpu_memory_bytes()
    if memory_limit == 0:
        raise RuntimeError("Unable to determine GPU memory. Ensure NVIDIA drivers are correctly installed.")

    for page_idx, page in enumerate(pages):
        print(f"[INFO] Processing page {page_idx + 1}/{len(pages)}")
        page_tensor = torch.tensor(page).permute(2, 0, 1).float() / 255.0  # [C, H, W]

        # --- Step 1: DETECTION with dynamic scaling ---
        try:
            detection_dict, scale_used = detect_page_with_dynamic_scaling(
                page_tensor, det_model, memory_limit, min_scale=min_scale, scale_step=scale_step
            )
        except RuntimeError as e:
            print(f"[ERROR] Failed to detect text on page {page_idx + 1}: {e}")
            continue

        preds = detection_dict["preds"][0] if len(detection_dict["preds"]) > 0 else None

        if not preds or len(preds["words"]) == 0:
            print(f"[INFO] No text detected on page {page_idx + 1}")
            continue

        page_boxes = preds["words"]  # Assuming relative [0..1] coordinates

        # --- Step 2: RECOGNITION for each box ---
        cropped_images = []
        polygons_and_conf = []

        for box_idx, (xmin, ymin, xmax, ymax, conf) in enumerate(page_boxes):
            # Convert relative coordinates to absolute coordinates
            abs_xmin = xmin * full_width
            abs_ymin = ymin * full_height
            abs_xmax = xmax * full_width
            abs_ymax = ymax * full_height

            # Crop from the original image
            crop = full_img.crop((abs_xmin, abs_ymin, abs_xmax, abs_ymax))

            # Resize for TrOCR input (384x384 is standard)
            crop_resized = crop.resize((384, 384), Image.Resampling.LANCZOS)
            cropped_images.append(crop_resized)

            # Keep track of polygon and confidence
            polygon = [
                [abs_xmin, abs_ymin],
                [abs_xmax, abs_ymin],
                [abs_xmax, abs_ymax],
                [abs_xmin, abs_ymax],
            ]
            polygons_and_conf.append((polygon, float(conf)))

        if not cropped_images:
            continue

        # --- Step 3: Recognition with adaptive chunking ---
        crop_tensors = processor(images=cropped_images, return_tensors="pt", padding=True).pixel_values.to(device)

        def generate_fn(batch):
            return trocr_model.generate(batch, max_length=40)  # Adjust max_length as needed

        try:
            # Recalculate available memory before recognition
            # For simplicity, using the same memory_limit. Alternatively, retrieve updated memory.
            generated_ids = adaptive_chunking(crop_tensors, generate_fn, memory_limit)
            if generated_ids is None:
                print(f"[WARNING] No recognition results for page {page_idx + 1}")
                continue
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        except RuntimeError as e:
            print(f"[ERROR] Failed during recognition on page {page_idx + 1}: {e}")
            continue

        # --- Step 4: Compile OCR results ---
        for (polygon, conf), txt in zip(polygons_and_conf, texts):
            final_ocr_results.append([polygon, [txt, conf]])

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
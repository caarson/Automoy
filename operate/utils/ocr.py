import os
import json
import subprocess
import wmi
import pynvml
import ctypes
from datetime import datetime
from PIL import Image, ImageDraw
import easyocr
import torch

try:
    from operate.config import Config
    config = Config()
except ImportError:
    class DummyConfig:
        verbose = True
    config = DummyConfig()

###############################################################
# Force VRAM usage up to ~4GB, then if more is needed,
# Windows may push some occupant data into Shared Memory.
#
# Implementation:
# 1) Load the entire EasyOCR model.
# 2) After model load, measure how much VRAM is used.
#    Then allocate a filler 'occupant' tensor on the GPU
#    to fill the remainder up to ~3.5GB or so, leaving ~0.5GB overhead.
# 3) If occupant alloc fails, fallback to smaller occupant.
# 4) Then run OCR in inference_mode.
#
# This ensures the GPU tries to use all 4GB VRAM, pushing occupant data
# or image data to Shared Memory as needed. It's not guaranteed to speed
# up OCR, but it does forcibly fill VRAM.
#
# Potential OOM risk: If the images are large, you may still see OOM.
# If that happens, reduce occupant_fallback or skip occupant entirely.
###############################################################

torch.backends.cudnn.benchmark = True  # Speed up conv

BASE_DIR = os.path.dirname(__file__)
OCR_DIR = os.path.join(BASE_DIR, "..", "ocr")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "..", "screenshots", "screenshot.png")
os.makedirs(OCR_DIR, exist_ok=True)

########################################
# GPU & Shared Memory Info (just prints)
########################################

def get_dedicated_memory_from_nvidia_smi():
    try:
        import subprocess
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"
        ], encoding="utf-8")
        dedicated_memory_mb = int(output.strip())
        return dedicated_memory_mb / 1024.0
    except Exception as e:
        print(f"[ERROR] nvidia-smi fetch error: {e}")
        return 0

def get_free_shared_memory(fallback_shared_gb=6.0, max_shared_gb=15.9, tolerance_gb=0.5):
    """Interpret free system RAM as 'shared GPU memory' on Windows."""
    import wmi
    try:
        w = wmi.WMI(namespace="root\CIMV2")
        os_info = w.query("SELECT FreePhysicalMemory FROM Win32_OperatingSystem")
        if os_info and len(os_info) > 0:
            free_kb = int(os_info[0].FreePhysicalMemory)
            free_gb = free_kb / (1024**2)
            if free_gb > max_shared_gb:
                print(f"[WARNING] Shared memory reported unusually high: {free_gb:.2f} GB. Clamping to {max_shared_gb}.")
                return max_shared_gb
            elif free_gb < fallback_shared_gb - tolerance_gb:
                print(f"[WARNING] Shared memory reported unusually low: {free_gb:.2f} GB. Using fallback {fallback_shared_gb}.")
                return fallback_shared_gb
            return free_gb
        print(f"[WARNING] No valid data for FreePhysicalMemory. Using fallback {fallback_shared_gb}.")
        return fallback_shared_gb
    except Exception as e:
        print(f"[ERROR] Could not retrieve free shared memory: {e}")
        return fallback_shared_gb

def get_gpu_memory_task_manager():
    try:
        import wmi
        w = wmi.WMI(namespace="root\CIMV2")
        video_controllers = w.query("SELECT AdapterRAM, Name, DriverVersion FROM Win32_VideoController")
        for vc in video_controllers:
            if "NVIDIA" in vc.Name:
                adapter_ram = vc.AdapterRAM

                # If adapter_ram is None or negative, fallback to nvidia-smi
                if not adapter_ram or adapter_ram <= 0:
                    dedicated_memory_gb = get_dedicated_memory_from_nvidia_smi()
                else:
                    dedicated_memory_gb = adapter_ram / (1024**3)

                # If somehow negative, clamp to 0
                if dedicated_memory_gb < 0:
                    dedicated_memory_gb = 0

                # Also retrieve shared memory
                shared_memory_gb = get_free_shared_memory()

                print(f"Adapter: {vc.Name}")
                print(f"Driver Version: {vc.DriverVersion}")
                print(f"Dedicated Memory: {dedicated_memory_gb:.2f} GB")
                print(f"Shared Memory (Free): {shared_memory_gb:.2f} GB")
                return dedicated_memory_gb, shared_memory_gb
        print("No NVIDIA GPU found.")
        return 0, 0
    except Exception as e:
        print(f"[ERROR] GPU info error: {e}")
        return 0, 0
    except Exception as e:
        print(f"[ERROR] Error retrieving GPU memory: {e}")
        return 0, 0
    except Exception as e:
        print(f"[ERROR] GPU info error: {e}")
        return 0

########################################
# The occupant logic to fill VRAM
########################################

def occupy_vram_up_to_4gb(overhead_gb=0.5):
    """Fill leftover VRAM up to ~4GB minus overhead, with a single occupant tensor."""
    used_gb = torch.cuda.memory_allocated() / (1024**3)
    occupant_gb = 4.0 - overhead_gb - used_gb
    if occupant_gb <= 0.0:
        print(f"[INFO] No occupant needed, used VRAM already ~{used_gb:.2f} GB.")
        return None

    occupant_numel = int((occupant_gb * (1024**3)) / 4)  # float32 => 4 bytes each
    try:
        occupant = torch.empty(occupant_numel, dtype=torch.float32, device='cuda')
        occupant.fill_(1.0)
        torch.cuda.synchronize()
        print(f"[INFO] Occupying ~{occupant_gb:.2f} GB of VRAM.")
        return occupant
    except RuntimeError as e:
        print(f"[WARNING] occupant allocation of ~{occupant_gb:.2f} GB failed: {str(e)}")
        return None

########################################
# OCR Implementation
########################################

def perform_easyocr(image_path, gpu=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    adapter_ram_gb = get_gpu_memory_task_manager()

    print("\n[INFO] Loading EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=gpu, download_enabled=True)

    print("[INFO] VRAM usage after model load:", f"{torch.cuda.memory_allocated()/(1024**2):.2f} MB")

    occupant = None
    # We'll attempt a couple occupant sizes if the first fails.
    occupant_fallbacks = [0.5, 1.0]
    # overhead_gb means how many GB to leave free in the 4GB VRAM

    for overhead in occupant_fallbacks:
        occupant = occupy_vram_up_to_4gb(overhead_gb=overhead)
        if occupant is not None:
            break

    print("[INFO] Running OCR now (inference_mode)...")
    with torch.inference_mode():
        results = reader.readtext(
            image_path,
            width_ths=0.95,
            height_ths=0.95,
            slope_ths=0.5,
            contrast_ths=0.5
        )

    # Release occupant
    if occupant is not None:
        del occupant
        torch.cuda.empty_cache()

    return results

#####################################
# Store OCR results
#####################################

def store_ocr_results(ocr_data, output_path):
    def safe_json(obj):
        if isinstance(obj, (int, float, str, bool, list, dict)) or obj is None:
            return obj
        if hasattr(obj, "item"):
            return obj.item()
        return float(obj) if isinstance(obj, (int, float)) else str(obj)

    safe_ocr_data = []
    for entry in ocr_data:
        box = entry[0]
        text = entry[1]
        conf = entry[2] if len(entry) > 2 else None
        box_converted = [[float(pt[0]), float(pt[1])] for pt in box]
        safe_entry = [box_converted, safe_json(text), safe_json(conf)]
        safe_ocr_data.append(safe_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_ocr_data, f, ensure_ascii=False, indent=4)
    print(f"OCR data saved at: {output_path}")

def store_ocr_text(ocr_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ocr_data:
            f.write(f"{item[1]}\n")
    print(f"OCR text saved at: {output_path}")

if __name__ == "__main__":
    SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "..", "screenshots", "screenshot.png")
    if not os.path.exists(SCREENSHOT_PATH):
        print(f"Screenshot not found at {SCREENSHOT_PATH}.")
    else:
        try:
            print("\nPerforming OCR on:", SCREENSHOT_PATH)
            ocr_results = perform_easyocr(SCREENSHOT_PATH, gpu=True)

            if ocr_results:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ocr_text_path = os.path.join(OCR_DIR, f"ocr_text_{ts}.txt")
                store_ocr_text(ocr_results, ocr_text_path)

                ocr_data_path = os.path.join(OCR_DIR, f"ocr_data_{ts}.json")
                store_ocr_results(ocr_results, ocr_data_path)
                print(f"OCR data saved at: {ocr_data_path}")
            else:
                print("No OCR results to save.")
        except Exception as e:
            print(f"Error: {e}")

import os
import json
from datetime import datetime
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.transforms import functional as F  # For resizing and padding

# --- docTR imports for DETECTION ---
from doctr.io import DocumentFile
from doctr.models import detection

# --- TrOCR imports for RECOGNITION ---
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# --- Local config import, if you have one ---
from operate.config import Config

# Load configuration
config = Config()

# Dynamic path setup
BASE_DIR = os.path.dirname(__file__)
OCR_DIR = os.path.join(BASE_DIR, "..", "ocr")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "..", "screenshots", "screenshot.png")

# Ensure OCR directory exists
os.makedirs(OCR_DIR, exist_ok=True)

# Configure CUDA to use shared memory and limit VRAM to 2GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"  # Manage fragmentation
torch.cuda.set_per_process_memory_fraction(1.0, device=torch.device('cuda:0'))  # Reserve 50% of 4GB VRAM (2GB limit)

# -----------------------------
# 1) Initialize DETECTION (docTR)
# -----------------------------
det_model = detection.db_resnet50(pretrained=True).eval()

# -----------------------------
# 2) Initialize RECOGNITION (TrOCR)
# -----------------------------
trocr_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(trocr_name)
trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_name).eval()

# Move models to GPU
det_model = det_model.cuda()
trocr_model = trocr_model.cuda()


def perform_ocr(image_path):
    """
    End-to-end OCR using docTR for detection and TrOCR for recognition.
    Processes the input image under a 6GB memory limit (2GB VRAM + 4GB shared memory).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the image as a tensor
    pages = DocumentFile.from_images(image_path)  # Returns a list of NumPy arrays
    pages_tensors = []
    for idx, page in enumerate(pages):
        tensor = torch.tensor(page).permute(2, 0, 1).float() / 255  # Convert to NCHW

        # Ensure dimensions are divisible by 32 (stride of the detection model)
        height, width = tensor.shape[1:]
        padded_height = (height + 31) // 32 * 32
        padded_width = (width + 31) // 32 * 32
        padding = [0, padded_width - width, 0, padded_height - height]  # Pad (left, right, top, bottom)

        padded_tensor = F.pad(tensor, padding, fill=0)
        pages_tensors.append(padded_tensor)
        print(f"[DEBUG] Page {idx + 1} padded to {padded_tensor.shape}")

    # Stack tensors into a batch
    pages_tensors = torch.stack(pages_tensors)
    print(f"[DEBUG] Stacked tensor shape: {pages_tensors.shape}")

    # Move to GPU (explicitly use shared memory)
    pages_tensors = pages_tensors.cuda()

    # Perform text detection
    detection_results = det_model(pages_tensors)
    print(f"[DEBUG] Detection results obtained.")

    if not detection_results['preds']:
        raise ValueError("No text detected in the image.")

    page_boxes = detection_results['preds'][0]['words']
    final_ocr_results = []

    # Open original image
    with Image.open(image_path) as img:
        width, height = img.size

        # Process bounding boxes
        resized_crops = []
        for idx, box in enumerate(page_boxes):
            print(f"[DEBUG] Processing box {idx + 1}/{len(page_boxes)}")
            xmin, ymin, xmax, ymax, confidence = box
            abs_xmin, abs_ymin = xmin * width, ymin * height
            abs_xmax, abs_ymax = xmax * width, ymax * height

            # Crop region from the original image
            crop = img.crop((abs_xmin, abs_ymin, abs_xmax, abs_ymax))

            # Resize the crop while maintaining aspect ratio
            crop_resized = crop.resize((384, 384), Image.Resampling.LANCZOS)
            resized_crops.append(crop_resized)

        # Perform batch recognition
        crop_tensors = processor(images=resized_crops, return_tensors="pt", padding=True).pixel_values
        crop_tensors = crop_tensors.cuda()

        generated_ids = trocr_model.generate(crop_tensors)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for idx, box in enumerate(page_boxes):
            xmin, ymin, xmax, ymax, confidence = box
            abs_xmin, abs_ymin = xmin * width, ymin * height
            abs_xmax, abs_ymax = xmax * width, ymax * height

            # Build polygon
            polygon = [
                [abs_xmin, abs_ymin],
                [abs_xmax, abs_ymin],
                [abs_xmax, abs_ymax],
                [abs_xmin, abs_ymax],
            ]
            final_ocr_results.append([polygon, [texts[idx], float(confidence)]])

    return final_ocr_results


def store_ocr_results(ocr_data, output_path):
    """
    Store the entire OCR output as JSON in the specified file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=4)


# Test example
if __name__ == "__main__":
    if not os.path.exists(SCREENSHOT_PATH):
        print(f"Screenshot not found at {SCREENSHOT_PATH}. Please provide a valid image.")
    else:
        try:
            print(f"Performing OCR on: {SCREENSHOT_PATH}")
            ocr_results = perform_ocr(SCREENSHOT_PATH)

            # Save raw OCR data
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_data_path = os.path.join(OCR_DIR, f"ocr_data_{datetime_str}.json")
            store_ocr_results(ocr_results, ocr_data_path)
            print(f"OCR data saved at: {ocr_data_path}")

        except Exception as e:
            print(f"Error: {e}")

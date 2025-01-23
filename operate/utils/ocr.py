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

# Configure CUDA shared memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

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

# Move models to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_model = det_model.to(device)
trocr_model = trocr_model.to(device)


def adaptive_chunking(tensors, model_or_generate_fn, memory_limit, chunk_size=1):
    """
    Process tensors dynamically in chunks, using all available memory but avoiding OOM errors.
    :param tensors: a batch tensor on the correct device
    :param model_or_generate_fn: either a model.forward or model.generate function
    :param memory_limit: approximate upper bound on GPU memory usage
    :param chunk_size: initial chunk size
    """
    results = []
    i = 0
    # If the model_or_generate_fn is a standard forward, we expect model_or_generate_fn(tensor).
    # If it is .generate (for huggingface), it returns a list/tensor of IDs.
    
    while i < tensors.size(0):
        try:
            chunk = tensors[i : i + chunk_size]
            # Quick approximation to see if we *might* exceed memory
            # (This is a rough check because forward/backward memory usage includes activations.)
            memory_required = chunk.element_size() * chunk.nelement()
            if (torch.cuda.memory_allocated() + memory_required) > memory_limit:
                # Force an OOM-like exception so we can handle chunk_size reduction
                raise RuntimeError("Memory limit exceeded. Reducing chunk size.")
            
            # Actually run the model
            output = model_or_generate_fn(chunk)
            
            # If it's a generate() call, just collect the IDs
            # If it's a forward, collect the output logits.
            # We'll unify them below.
            if isinstance(output, torch.Tensor):
                results.append(output)
            else:
                # Typically output from generate is a Tensor of token IDs
                # Could also be a list of Tensors. Adjust as needed.
                results.append(output)
            
            i += chunk_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] Chunk {i} caused OOM. Reducing chunk size from {chunk_size} to {max(1, chunk_size // 2)}")
                chunk_size = max(1, chunk_size // 2)
                if chunk_size == 1:
                    # Try a forced GPU cache clear to see if we can proceed.
                    torch.cuda.empty_cache()
            else:
                raise

    # Concatenate results if they're Tensors. If they are lists of Tensors (e.g. from .generate), do a simple cat.
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=0)
    else:
        # If the model's output is a list of Tensors (like token ID sequences),
        # we can flatten them. Usually for huggingface generate: results is [Tensor, Tensor, ...].
        return torch.cat(results, dim=0)


def detect_page_with_fallback(page_tensor, model, memory_limit, min_scale=0.1):
    """
    Attempt to run docTR detection on a single page tensor. If OOM occurs,
    downscale the page and try again, until it fits or we reach min_scale.
    
    :param page_tensor: CHW float tensor on CPU (initially).
    :param model: docTR detection model (on GPU).
    :param memory_limit: approximate memory limit in bytes
    :param min_scale: lowest allowed scale factor before we give up
    :return: detection_results (dictionary), scale_factor actually used
    """
    scale_factor = 1.0
    page_height, page_width = page_tensor.shape[1], page_tensor.shape[2]
    
    while True:
        # Resize if scale < 1.0
        if scale_factor < 1.0:
            new_h = int(page_height * scale_factor)
            new_w = int(page_width * scale_factor)
            if new_h < 1 or new_w < 1:
                raise RuntimeError("Image too large to process even at minimum scale.")

            scaled_page = F.resize(page_tensor, (new_h, new_w))
        else:
            scaled_page = page_tensor
        
        # Make sure dims divisible by 32 for DB/ResNet-based models (recommended, though not strictly mandatory).
        # We'll do a small pad to keep docTR happy.
        h, w = scaled_page.shape[1], scaled_page.shape[2]
        padded_h = (h + 31) // 32 * 32
        padded_w = (w + 31) // 32 * 32
        padding = [0, padded_w - w, 0, padded_h - h]  # left, right, top, bottom
        scaled_page_padded = F.pad(scaled_page, padding, fill=0)
        
        # docTR expects a list of NP arrays in shape HWC or a batch of Tensors in shape BCHW
        # We'll pass a batch of size 1 in torch. docTR can handle that.
        input_batch = scaled_page_padded.unsqueeze(0).to(device)
        
        try:
            # Attempt detection
            # docTR detection models accept either np.array(list of) or Torch Tensors.
            # If you want to feed Torch Tensors directly, ensure the docTR model is set to 'eval' & on GPU.
            detection_results = model(input_batch)
            return detection_results, scale_factor

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARNING] OOM at scale_factor={scale_factor:.2f}. Halving scale.")
                scale_factor *= 0.5
                torch.cuda.empty_cache()
                if scale_factor < min_scale:
                    raise RuntimeError(
                        "Cannot reduce scale further; not enough GPU memory to process the image."
                    ) from e
            else:
                raise


def perform_ocr(image_path, memory_limit, initial_chunk_size=1):
    """
    End-to-end OCR using docTR for detection and TrOCR for recognition.
    Dynamically adjusts processing to avoid exceeding memory limits.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the image as a docTR DocumentFile object (list of pages as NumPy arrays)
    pages = DocumentFile.from_images(image_path)  # Returns list of np.array([...])
    
    final_ocr_results = []
    
    with Image.open(image_path) as full_img:
        full_width, full_height = full_img.size

    for page_idx, page in enumerate(pages):
        # Convert page to CHW float (on CPU for now)
        page_tensor = torch.tensor(page).permute(2, 0, 1).float() / 255.0  # [C,H,W]

        # 1) Run detection with fallback scaling
        detection_dict, scale_used = detect_page_with_fallback(page_tensor, det_model, memory_limit)
        preds = detection_dict["preds"][0] if len(detection_dict["preds"]) > 0 else None
        
        if not preds or len(preds["words"]) == 0:
            print(f"[INFO] No text detected on page {page_idx + 1}")
            continue
        
        # 2) For each bounding box, scale it up to the *original* page size
        # docTR returns relative coords in [0, 1] or absolute in [0, scaled_size].
        # Usually docTR's DB is in relative [0..1], but confirm your version. We'll assume [0..1].
        # We then multiply by the original image dimension. BUT if docTR used a smaller scale,
        # the model’s built-in transforms typically still produce relative coordinates
        # in terms of the input size. So we only need to map them to the real full_img size.

        # The detection model’s default behavior is to produce boxes in relative coordinates (0..1),
        # so scaling the final box to the *true* full_img dimension is just [x_rel*w_original, y_rel*h_original].
        page_boxes = preds["words"]
        
        # 3) Prepare crops for recognition
        cropped_images = []
        polygons_and_confidence = []
        
        with Image.open(image_path) as full_img:            
            for box_idx, (xmin, ymin, xmax, ymax, confidence) in enumerate(page_boxes):
                # docTR is typically in [0..1], so:
                abs_xmin = xmin * full_width
                abs_ymin = ymin * full_height
                abs_xmax = xmax * full_width
                abs_ymax = ymax * full_height

                # Crop from the original image
                # If your docTR model is giving absolute coords in the scaled image,
                # you'd do an extra step to scale from scaled image -> full image.
                crop = full_img.crop((abs_xmin, abs_ymin, abs_xmax, abs_ymax))

                # Resize for TrOCR input. E.g. 384x384 is typical for base-stage1
                crop_resized = crop.resize((384, 384), Image.Resampling.LANCZOS)
                cropped_images.append(crop_resized)
                
                # Keep track of polygon + confidence for final JSON
                # We'll store them after we get text from recognition
                polygon = [
                    [abs_xmin, abs_ymin],
                    [abs_xmax, abs_ymin],
                    [abs_xmax, abs_ymax],
                    [abs_xmin, abs_ymax],
                ]
                polygons_and_confidence.append((polygon, confidence))

        if not cropped_images:
            continue
        
        # 4) Recognition in chunks
        crop_tensors = processor(images=cropped_images, return_tensors="pt", padding=True).pixel_values.to(device)
        
        # model.generate needs to be passed as a callable
        # partial function: e.g. lambda x: trocr_model.generate(x, ...optional args...)
        def generate_fn(batch):
            return trocr_model.generate(batch)
        
        generated_ids = adaptive_chunking(crop_tensors, generate_fn, memory_limit, initial_chunk_size)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Combine results
        for (polygon, conf), txt in zip(polygons_and_confidence, texts):
            final_ocr_results.append([polygon, [txt, float(conf)]])

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
            # Example memory limit (6GB).
            # Combine GPU VRAM + system shared memory if you want a total approximate limit in bytes.
            memory_limit = 6 * 1024**3

            print(f"Performing OCR on: {SCREENSHOT_PATH}")
            ocr_results = perform_ocr(SCREENSHOT_PATH, memory_limit)

            # Save raw OCR data
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_data_path = os.path.join(OCR_DIR, f"ocr_data_{datetime_str}.json")
            store_ocr_results(ocr_results, ocr_data_path)
            print(f"OCR data saved at: {ocr_data_path}")

        except Exception as e:
            print(f"Error: {e}")

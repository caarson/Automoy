import os
import sys

# Dynamically add the project root to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from operate.utils.ocr import perform_easyocr as perform_ocr
from operate.utils.yolo import YOLODetector
from operate.utils.screenshot import capture_screen_with_cursor

# Initialize YOLO detector
yolo_detector = YOLODetector()

def simplify_list(lst, label, max_items=10):
    """
    Helper to join a list of strings with a label. If there are more than
    max_items, only show the first few followed by a summary count.
    """
    total = len(lst)
    if total == 0:
        return f"{label} (0): None"
    if total <= max_items:
        items = " | ".join(lst)
    else:
        items = " | ".join(lst[:max_items]) + f" | ... ({total - max_items} more)"
    return f"{label} ({total}): {items}"

def simplify_preprocessed_results(data):
    """
    Convert the preprocessed OCR and YOLO data into a concise summary string.
    """
    matched_summary = simplify_list(data.get("matched_results", []), "Matched Results")
    ocr_summary = simplify_list(data.get("ocr_results", []), "OCR Results")
    yolo_summary = simplify_list(data.get("yolo_results", []), "YOLO Results")
    summary = (
        f"Preprocessed Data Summary:\n\n"
        f"{matched_summary}\n\n"
        f"{ocr_summary}\n\n"
        f"{yolo_summary}"
    )
    return summary

async def preprocess_with_ocr_and_yolo(screenshot_path):
    """
    Preprocess the screen using OCR and YOLO to extract relevant UI elements and text.

    Returns:
        tuple (summary_string, full_data_dict):
            - summary_string: A short, human-readable summary of matched, OCR, and YOLO results.
            - full_data_dict: A dictionary containing full lists of matched_results, ocr_results, and yolo_results.
    """
    print("[preprocessing] Performing OCR and YOLO preprocessing...")

    # Perform OCR to extract text and coordinates
    ocr_results = perform_ocr(screenshot_path)
    print(f"[preprocessing] OCR Results: {ocr_results}")

    # Perform YOLO detection to identify UI elements
    yolo_results = yolo_detector.detect_objects(screenshot_path)
    print(f"[preprocessing] YOLO Results: {yolo_results}")

    # Combine OCR and YOLO results based on overlapping coordinates
    matched_results = []
    for yolo_obj in yolo_results:
        yolo_x, yolo_y = yolo_obj["x"], yolo_obj["y"]
        matched = False
        for ocr_obj in ocr_results:
            polygon, text = ocr_obj[0], ocr_obj[1]
            # Match if any point in the OCR polygon is near the YOLO center
            if any(abs(p[0] / 1920 - yolo_x) < 0.1 and abs(p[1] / 1080 - yolo_y) < 0.1 for p in polygon):
                if text:
                    matched_results.append(f"text: {text} @ ({yolo_x}, {yolo_y})")
                else:
                    matched_results.append(f"button @ ({yolo_x}, {yolo_y})")
                matched = True
                break
        if not matched:
            matched_results.append(f"button @ ({yolo_x}, {yolo_y})")

    # Simplify OCR results: if text exists, include text and first coordinate; otherwise mark as button.
    ocr_simplified = [
        f"text: {obj[1]} @ ({obj[0][0][0]}, {obj[0][0][1]})" if obj[1]
        else f"button @ ({obj[0][0][0]}, {obj[0][0][1]})"
        for obj in ocr_results
    ]
    # Simplify YOLO results: simply output button coordinates.
    yolo_simplified = [
        f"button @ ({obj['x']}, {obj['y']})"
        for obj in yolo_results
    ]

    print(f"[preprocessing] Matched Results: {matched_results}")
    print(f"[preprocessing] OCR Results: {ocr_simplified}")
    print(f"[preprocessing] YOLO Results: {yolo_simplified}")
    
    # Build the full data dictionary.
    full_data = {
        "matched_results": matched_results,
        "ocr_results": ocr_simplified,
        "yolo_results": yolo_simplified
    }
    
    # Create a short summary string to include in the initial prompt.
    summary_string = (
        f"Preprocessed Data Summary:\n"
        f"Matched: {len(matched_results)} items\n"
        f"OCR: {len(ocr_simplified)} items\n"
        f"YOLO: {len(yolo_simplified)} items\n"
        "If you need additional details on OCR or YOLO, please request them."
    )

    # Return both the short summary and the full data
    return summary_string, full_data

# Main script for testing
if __name__ == "__main__":
    import asyncio

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    screenshots_dir = os.path.join(project_dir, "screenshots")
    screenshot_path = os.path.join(screenshots_dir, "screenshot.png")

    os.makedirs(screenshots_dir, exist_ok=True)

    print("[preprocessing] Capturing screenshot...")
    capture_screen_with_cursor(screenshot_path)

    async def main():
        if os.path.exists(screenshot_path):
            print(f"[preprocessing] Screenshot saved to: {screenshot_path}")
            results = await preprocess_with_ocr_and_yolo(screenshot_path)
            print("[preprocessing] Final Simplified Results:")
            print(results)
        else:
            print(f"[ERROR] Screenshot not found at {screenshot_path}.")

    print("[preprocessing] Starting preprocessing pipeline...")
    asyncio.run(main())

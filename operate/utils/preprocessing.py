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

async def preprocess_with_ocr_and_yolo(screenshot_path):
    """
    Preprocess the screen using OCR and YOLO to extract relevant UI elements and text.
    Args:
        screenshot_path (str): Path to the screenshot for processing.
    Returns:
        dict: Simplified matched, OCR, and YOLO results with only necessary content and coordinates.
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
            
            # Match YOLO and OCR results if coordinates are similar
            if any(
                abs(p[0] / 1920 - yolo_x) < 0.1 and abs(p[1] / 1080 - yolo_y) < 0.1
                for p in polygon
            ):
                if text:
                    matched_results.append(f"{{text, {text}, {yolo_x} {yolo_y}}}")
                else:
                    matched_results.append(f"{{button, {yolo_x} {yolo_y}}}")
                matched = True
                break  # Stop checking other OCR polygons for this YOLO object
    
        if not matched:
            matched_results.append(f"{{button, {yolo_x} {yolo_y}}}")

    # Simplified OCR and YOLO results
    ocr_simplified = [
        f"{{text, {obj[1]}, {obj[0][0][0]} {obj[0][0][1]}}}" if obj[1] else f"{{button, {obj[0][0][0]} {obj[0][0][1]}}}"
        for obj in ocr_results
    ]
    yolo_simplified = [
        f"{{button, {obj['x']} {obj['y']}}}"
        for obj in yolo_results
    ]

    print(f"[preprocessing] Matched Results: {matched_results}")
    print(f"[preprocessing] OCR Results: {ocr_simplified}")
    print(f"[preprocessing] YOLO Results: {yolo_simplified}")
    
    return {
        "matched_results": matched_results,
        "ocr_results": ocr_simplified,
        "yolo_results": yolo_simplified
    }

# Main script
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
            print("[preprocessing] Final Results:")
            print(results)
        else:
            print(f"[ERROR] Screenshot not found at {screenshot_path}.")

    print("[preprocessing] Starting preprocessing pipeline...")
    asyncio.run(main())

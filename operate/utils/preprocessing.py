import os
import sys

# Dynamically add the project root to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from operate.utils.ocr import perform_ocr
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
        dict: Combined results from OCR and YOLO detection.
    """
    print("[preprocessing] Performing OCR and YOLO preprocessing...")

    # Perform OCR to extract text and coordinates
    ocr_results = perform_ocr(screenshot_path)
    print(f"[preprocessing] OCR Results: {ocr_results}")

    # Perform YOLO detection to identify UI elements
    yolo_results = yolo_detector.detect_objects(screenshot_path)
    print(f"[preprocessing] YOLO Results: {yolo_results}")

    # Combine OCR and YOLO results based on overlapping coordinates
    combined_results = []
    for yolo_obj in yolo_results:
        yolo_x, yolo_y = yolo_obj["x"], yolo_obj["y"]
        for ocr_obj in ocr_results:
            polygon, (text, confidence) = ocr_obj  # OCR result format
            # Match YOLO and OCR results if the coordinates are similar
            if any(
                abs(p[0] - yolo_x) < 0.05 and abs(p[1] - yolo_y) < 0.05
                for p in polygon
            ):
                combined_results.append({
                    "label": yolo_obj["label"],
                    "confidence": yolo_obj["confidence"],
                    "text": text,
                    "ocr_confidence": confidence,
                    "x": yolo_x,
                    "y": yolo_y
                })

    print(f"[preprocessing] Combined Results: {combined_results}")
    return combined_results


# Main script
if __name__ == "__main__":
    import asyncio

    # Dynamically determine the project root and screenshots directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    screenshots_dir = os.path.join(project_dir, "screenshots")
    screenshot_path = os.path.join(screenshots_dir, "screenshot.png")

    # Ensure the screenshot directory exists
    os.makedirs(screenshots_dir, exist_ok=True)

    # Capture a screenshot with the cursor
    print("[preprocessing] Capturing screenshot...")
    capture_screen_with_cursor(screenshot_path)

    # Perform preprocessing
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

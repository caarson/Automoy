import os
import sys

# Dynamically add the project root to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from operate.utils.ocr import perform_easyocr as perform_ocr  # Ensure the correct function name is used
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
        list: Simplified combined results from OCR and YOLO detection.
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
        matched = False
        for ocr_obj in ocr_results:
            polygon, text, confidence = ocr_obj[0], ocr_obj[1], ocr_obj[2]  # Adjusted format to handle tuple indexing
            # Debug: Print coordinate details for matching
            print(f"Matching YOLO: ({yolo_x}, {yolo_y}) with OCR Polygon: {polygon}")
            
            # Match YOLO and OCR results if the coordinates are similar
            if any(
                abs(p[0] / 1920 - yolo_x) < 0.1 and abs(p[1] / 1080 - yolo_y) < 0.1  # Normalize for screen resolution
                for p in polygon
            ):
                combined_results.append({
                    "label": yolo_obj["label"],
                    "confidence": yolo_obj["confidence"],
                    "text": text,
                    "ocr_confidence": confidence,
                    "coordinates": {
                        "x": yolo_x,
                        "y": yolo_y
                    },
                    "matched": True
                })
                matched = True
                break  # Stop checking other OCR polygons for this YOLO object
    
        if not matched:
            # Add unmatched YOLO objects for debugging
            combined_results.append({
                "label": yolo_obj["label"],
                "confidence": yolo_obj["confidence"],
                "text": None,
                "ocr_confidence": None,
                "coordinates": {
                    "x": yolo_x,
                    "y": yolo_y
                },
                "matched": False
            })

    # Add unmatched OCR results for debugging
    for ocr_obj in ocr_results:
        polygon, text, confidence = ocr_obj[0], ocr_obj[1], ocr_obj[2]
        print(f"Unmatched OCR Polygon: {polygon}, Text: {text}, Confidence: {confidence}")

    # Simplify results for display
    simplified_results = [
        {
            "Text": result.get("text", "N/A"),
            "Label": result.get("label", "N/A"),
            "Confidence": result.get("confidence", 0),
            "OCR Confidence": result.get("ocr_confidence", 0),
            "Coordinates": result.get("coordinates", {}),
            "Matched": result.get("matched", False)
        }
        for result in combined_results
    ]

    print(f"[preprocessing] Combined Results: {simplified_results}")
    return simplified_results

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
            for result in results:
                print(result)
        else:
            print(f"[ERROR] Screenshot not found at {screenshot_path}.")

    print("[preprocessing] Starting preprocessing pipeline...")
    asyncio.run(main())
from operate.utils.ocr import perform_ocr
from operate.utils.yolo import YOLODetector
from operate.utils.screenshot import capture_screen_with_cursor

# Initialize YOLO detector
yolo_detector = YOLODetector()

async def preprocess_with_ocr_and_yolo(screenshot_path="screenshots/screenshot.png"):
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

    # Combine OCR and YOLO results
    combined_results = []
    for yolo_obj in yolo_results:
        for ocr_obj in ocr_results:
            # Match OCR text to YOLO objects by overlapping coordinates
            if yolo_obj["x"] == ocr_obj["x"] and yolo_obj["y"] == ocr_obj["y"]:
                combined_results.append({
                    "label": yolo_obj["label"],
                    "confidence": yolo_obj["confidence"],
                    "text": ocr_obj["text"],
                    "x": yolo_obj["x"],
                    "y": yolo_obj["y"]
                })

    print(f"[preprocessing] Combined Results: {combined_results}")
    return combined_results

# Test script
if __name__ == "__main__":
    import asyncio

    # Ensure the screenshot directory exists
    screenshot_path = "screenshots/test_screenshot.png"
    capture_screen_with_cursor(screenshot_path)

    async def test_preprocessing():
        results = await preprocess_with_ocr_and_yolo(screenshot_path)
        print(f"Test Preprocessing Results: {results}")

    print("[Testing] Starting test for preprocess_with_ocr_and_yolo...")
    asyncio.run(test_preprocessing())

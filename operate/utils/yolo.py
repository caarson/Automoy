import torch
from ultralytics import YOLO
import pyautogui
import os

# Load configuration
weights_path = os.path.join("models", "weights", "best.pt")
screen_width, screen_height = pyautogui.size()

class YOLODetector:
    def __init__(self):
        """
        Initializes the YOLO detector and loads the model.
        """
        print("Initializing YOLO detector...")
        from operate.utils.check_cuda import check_cuda
        cuda_available = check_cuda()
        device = "cuda" if cuda_available else "cpu"

        # Load YOLO model and set device
        self.model = YOLO(weights_path)
        self.model.to(device)

        if cuda_available:
            print(f"CUDA enabled. Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available. Running on CPU.")

    def detect_objects(self, image_path):
        """
        Detects all objects in the image and returns their labels and coordinates.
        Args:
            image_path (str): Path to the image file.
        Returns:
            list: List of detected objects in the format:
                [{'label': 'button', 'confidence': 0.9, 'x': 0.5, 'y': 0.4}]
        """
        results = self.model(image_path)
        detected_objects = []

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box
                label = self.model.names[int(class_id)]

                # Convert YOLO bounding box to central point (percent of screen size)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                x_percent = round(x_center / screen_width, 3)
                y_percent = round(y_center / screen_height, 3)

                detected_objects.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "x": x_percent,
                    "y": y_percent
                })

        return detected_objects

# Test example
if __name__ == "__main__":
    yolo_detector = YOLODetector()
    test_image = "screenshots/screenshot.png"  # Ensure this path exists
    detections = yolo_detector.detect_objects(test_image)

    # Print all detected objects
    print("Detected objects:")
    for obj in detections:
        print(f"Label: {obj['label']}, Confidence: {obj['confidence']}, Coordinates: ({obj['x']}, {obj['y']})")

import torch
from ultralytics import YOLO
import pyautogui
import os

# Configuration
weights_path = os.path.join(os.path.dirname(__file__), "..", "data", "YOLO", "models", "socYOLO.pt")
screen_width, screen_height = pyautogui.size()

class YOLODetector:
    def __init__(self, weights=weights_path, device="cuda"):
        """
        Initializes the YOLO detector and loads the model.
        Args:
            weights (str): Path to the YOLO model weights file.
            device (str): The device to run YOLO on ("cuda" or "cpu").
        """
        print("Initializing YOLO detector...")

        # Load YOLO model and set device
        try:
            self.model = YOLO(weights)
            if torch.cuda.is_available() and device == "cuda":
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.model.to(self.device)

            print(f"YOLO model loaded with weights: {weights}")
            print(f"Running YOLO on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_objects(self, image_path):
        """
        Detects all objects in the image and returns their labels and coordinates.
        Args:
            image_path (str): Path to the image file.
        Returns:
            list: List of detected objects in the format:
                [{'label': 'button', 'confidence': 0.9, 'x': 0.5, 'y': 0.4}]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            results = self.model(image_path, device=self.device)
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
        except Exception as e:
            raise RuntimeError(f"Error during YOLO detection: {e}")

# Test example
if __name__ == "__main__":
    import sys

    # Set up paths and testing variables
    test_image = os.path.join(os.path.dirname(__file__), "..", "..", "screenshots", "screenshot.png")

    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}. Please provide a valid image.")
        sys.exit(1)

    # Initialize YOLO detector
    try:
        yolo_detector = YOLODetector()
        detections = yolo_detector.detect_objects(test_image)

        # Print all detected objects
        print("Detected objects:")
        for obj in detections:
            print(f"Label: {obj['label']}, Confidence: {obj['confidence']}, Coordinates: ({obj['x']}, {obj['y']})")
    except Exception as e:
        print(f"Error: {e}")

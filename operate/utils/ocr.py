from operate.config import Config
from PIL import Image, ImageDraw
import os
from datetime import datetime
import easyocr

# Load configuration
config = Config()

# Dynamic path setup
BASE_DIR = os.path.dirname(__file__)
OCR_DIR = os.path.join(BASE_DIR, "..", "ocr")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "..", "screenshots", "screenshot.png")

if not os.path.exists(OCR_DIR):
    os.makedirs(OCR_DIR)

def get_text_element(result, search_text, image_path):
    """
    Searches for a text element in the OCR results and returns its index. Also draws bounding boxes on the image.
    Args:
        result (list): The list of results returned by EasyOCR.
        search_text (str): The text to search for in the OCR results.
        image_path (str): Path to the original image.

    Returns:
        int: The index of the element containing the search text.

    Raises:
        Exception: If the text element is not found in the results.
    """
    if config.verbose:
        print("[get_text_element] Searching for:", search_text)

    # Open the original image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    found_index = None
    for index, element in enumerate(result):
        text = element[1]
        box = element[0]

        if config.verbose:
            # Draw bounding box in blue for all elements
            draw.polygon([tuple(point) for point in box], outline="blue")

        if search_text.lower() in text.lower():  # Case-insensitive match
            found_index = index
            if config.verbose:
                print(f"[get_text_element] Found '{search_text}' at index {index}")

    if found_index is not None:
        if config.verbose:
            # Draw bounding box of the found text in red
            box = result[found_index][0]
            draw.polygon([tuple(point) for point in box], outline="red")

            # Save the image with bounding boxes
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_image_path = os.path.join(OCR_DIR, f"ocr_image_{datetime_str}.png")
            image.save(ocr_image_path)
            print(f"[get_text_element] OCR image saved at: {ocr_image_path}")

        return found_index

    raise Exception("The text element was not found in the image")


def get_text_coordinates(result, index, image_path):
    """
    Gets the coordinates of the text element at the specified index as a percentage of screen width and height.
    Args:
        result (list): The list of results returned by EasyOCR.
        index (int): The index of the text element in the results list.
        image_path (str): Path to the screenshot image.

    Returns:
        dict: A dictionary containing the 'x' and 'y' coordinates as percentages of the screen width and height.
    """
    if index >= len(result):
        raise Exception("Index out of range in OCR results")

    # Get the bounding box of the text element
    bounding_box = result[index][0]

    # Calculate the center of the bounding box
    min_x = min([coord[0] for coord in bounding_box])
    max_x = max([coord[0] for coord in bounding_box])
    min_y = min([coord[1] for coord in bounding_box])
    max_y = max([coord[1] for coord in bounding_box])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Convert to percentages
    percent_x = round((center_x / width), 3)
    percent_y = round((center_y / height), 3)

    return {"x": percent_x, "y": percent_y}


# Test example
if __name__ == "__main__":
    # Initialize OCR reader
    reader = easyocr.Reader(["en"])

    if not os.path.exists(SCREENSHOT_PATH):
        print(f"Screenshot not found at {SCREENSHOT_PATH}. Please provide a valid image.")
    else:
        try:
            # Perform OCR
            print(f"Performing OCR on: {SCREENSHOT_PATH}")
            ocr_results = reader.readtext(SCREENSHOT_PATH)

            # Example: Search for specific text
            search_text = "example text"  # Replace with the text you want to search for
            text_index = get_text_element(ocr_results, search_text, SCREENSHOT_PATH)
            coordinates = get_text_coordinates(ocr_results, text_index, SCREENSHOT_PATH)

            print(f"Text '{search_text}' found at coordinates: {coordinates}")
        except Exception as e:
            print(f"Error: {e}")

import os
import platform
import subprocess
import pyautogui
from PIL import Image, ImageDraw, ImageGrab
import Xlib.display
import Xlib.X
import Xlib.Xutil  # not sure if Xutil is necessary


def capture_screen_with_cursor(file_path, bbox=None):
    user_platform = platform.system()

    if user_platform == "Windows":
        # Capture the specified area or the whole screen
        screenshot = pyautogui.screenshot(region=bbox)
        screenshot.save(file_path)
    elif user_platform == "Linux":
        if bbox:
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
        else:
            screen = Xlib.display.Display().screen()
            width, height = screen.width_in_pixels, screen.height_in_pixels
            x1, y1 = 0, 0
        # Grab the specified area using ImageGrab
        screenshot = ImageGrab.grab(bbox=(x1, y1, x1 + width, y1 + height))
        screenshot.save(file_path)
    elif user_platform == "Darwin":  # (Mac OS)
        if bbox:
            x1, y1, x2, y2 = bbox
            rect_str = f"{x1},{y1},{x2-x1},{y2-y1}"
            subprocess.run(["screencapture", "-R", rect_str, file_path])
        else:
            # Capture the full screen with the cursor
            subprocess.run(["screencapture", "-C", file_path])
    else:
        print(f"The platform you're using ({user_platform}) is not currently supported")

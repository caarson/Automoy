import os
import platform
import subprocess
import pyautogui
from PIL import Image, ImageDraw, ImageGrab
import Xlib.display
import Xlib.X
import Xlib.Xutil  # not sure if Xutil is necessary


def capture_screen_with_cursor(file_path, region=None):
    user_platform = platform.system()

    if user_platform == "Windows":
        # Capture the specified region or full screen if `region` is None
        if region:
            x1, y1, x2, y2 = region
            bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            bbox = None

        screenshot = pyautogui.screenshot(region=bbox)
        screenshot.save(file_path)
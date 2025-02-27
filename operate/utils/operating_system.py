import pyautogui
import platform
import time
import math

from operate.utils.misc import convert_percent_to_decimal

class OperatingSystem:
    def write(self, content):
        try:
            if not content:
                print("[OperatingSystem][write] No content provided to write.")
                return
            content = content.replace("\\n", "\n")
            pyautogui.write(content)
        except Exception as e:
            print("[OperatingSystem][write] Error:", e)

    def press(self, keys):
        try:
            if not keys or not isinstance(keys, list):
                print("[OperatingSystem][press] Invalid keys provided:", keys)
                return
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            print("[OperatingSystem][press] Error:", e)

    def mouse(self, click_detail, region=None):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x))
            y_pixel = int(screen_height * float(y))

            if region:
                x1, y1, x2, y2 = region
                if not (x1 <= x_pixel <= x2 and y1 <= y_pixel <= y2):
                    print(f"Click at ({x_pixel}, {y_pixel}) ignored (out of bounds).")
                    return  # Skip the action if out of bounds

            print(f"Mouse click at ({x_pixel}, {y_pixel}) within region.")
            pyautogui.click(x_pixel, y_pixel)

        except Exception as e:
            print("[OperatingSystem][mouse] Error:", e)

    def click_at_percentage(
        self,
        x_percentage,
        y_percentage,
        duration=0.2,
        circle_radius=50,
        circle_duration=0.5,
    ):
        try:
            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x_percentage))
            y_pixel = int(screen_height * float(y_percentage))

            pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

            start_time = time.time()
            while time.time() - start_time < circle_duration:
                angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
                x = x_pixel + math.cos(angle) * circle_radius
                y = y_pixel + math.sin(angle) * circle_radius
                pyautogui.moveTo(x, y, duration=0.1)

            pyautogui.click(x_pixel, y_pixel)
        except Exception as e:
            print("[OperatingSystem][click_at_percentage] Error:", e)

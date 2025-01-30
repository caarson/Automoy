import sys
import os
import time
import asyncio
from prompt_toolkit import prompt
from operate.exceptions import ModelNotRecognizedException
import platform

# from operate.models.prompts import USER_QUESTION, get_system_prompt
from operate.utils.prompts import (
    USER_QUESTION,
    get_system_prompt,
)
from operate.config import Config
from operate.utils.style import (
    ANSI_GREEN,
    ANSI_RESET,
    ANSI_YELLOW,
    ANSI_RED,
    ANSI_BRIGHT_MAGENTA,
    ANSI_BLUE,
    style,
)
from operate.utils.operating_system import OperatingSystem
from operate.model_handlers.handlers_apis import get_next_action
from operate.utils.screenshot import capture_screen_with_cursor

# Load configuration
config = Config()
operating_system = OperatingSystem()

def main(model, terminal_prompt, voice_mode=False, verbose_mode=False, define_region=False):
    """
    Main function for the Self-Operating Computer.

    Parameters:
    - model: The model used for generating responses.
    - terminal_prompt: A string representing the prompt provided in the terminal.
    - voice_mode: A boolean indicating whether to enable voice mode.

    Returns:
    None
    """

    # Check whether or not CUDA is enabled
    from operate.utils.check_cuda import check_cuda

    cuda_avaliable = check_cuda()

    print("CUDA STATUS:" + str(cuda_avaliable))

    if cuda_avaliable:
        print("CUDA enabled, proceeding with startup...")
    else:
        print("CUDA is either not installed or improperly configured or installed.\nExiting...")
        exit()

    mic = None
    # Initialize `WhisperMic`, if `voice_mode` is True

    config.verbose = verbose_mode
    config.validation(model)  # Fix: Removed extra argument

    # Initialize region_coords to avoid undefined variable error
    region_coords = None

    ## Boot Arguments:
    # Enable define a region boot arg
    if define_region:
        import tkinter as tk
        import threading
        from operate.utils.area_selector import select_area, create_outline_window

        done_event = threading.Event()
        region_coords = []  # Store the selected region coordinates

        def run_gui():
            root = tk.Tk()
            root.withdraw()

            def handle_selection(coords):
                nonlocal region_coords
                region_coords[:] = coords
                done_event.set()
                print(f"Selected region: {region_coords}")
                create_outline_window(region_coords, root)

            select_area(handle_selection)
            root.mainloop()

        gui_thread = threading.Thread(target=run_gui, daemon=True)
        gui_thread.start()

        done_event.wait()  # Wait for region selection
        print(f"Operating within region: {region_coords}")
    # Enable voice control
    if voice_mode:
        try:
            from whisper_mic import WhisperMic

            # Initialize WhisperMic if import is successful
            mic = WhisperMic()
        except ImportError:
            print(
                "Voice mode requires the 'whisper_mic' module. Please install it using 'pip install -r requirements-audio.txt'"
            )
            sys.exit(1)
    # Enable pass-through prompt
    if terminal_prompt:  # Skip objective prompt if it was given as an argument
        objective = terminal_prompt

    elif voice_mode:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RESET} Listening for your command... (speak now)"
        )
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Error in capturing voice input: {e}{ANSI_RESET}")
            return  # Exit if voice input fails
    else:
        print(
            f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n{USER_QUESTION}"
        )
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    system_prompt = get_system_prompt(model, objective)
    system_message = {"role": "system", "content": system_prompt}
    messages = [system_message]

    loop_count = 0

    session_id = None

    # Define screenshot path
    screenshot_path = os.path.join(os.getcwd(), "operate", "data", "screenshots", "screenshot.png")
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    capture_screen_with_cursor(screenshot_path)

    while True:
        if config.verbose:
            print("[Self Operating Computer] loop_count", loop_count)
        try:
            result = asyncio.run(get_next_action(model, messages, terminal_prompt, session_id, screenshot_path))
            if result is None:
                print(f"{ANSI_RED}[Error] get_next_action returned None.{ANSI_RESET}")
                break
            
            operations, session_id = result if isinstance(result, tuple) else ([], None)
            
            stop = operate(operations, model, region=region_coords)
            if stop:
                break

            loop_count += 1
            if loop_count > 10:
                break
        except ModelNotRecognizedException as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break
        except Exception as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break

import time

def operate(operations, model, region=None):
    # Function to check if a point is within the defined region
    def is_within_region(x, y, region):
        """Check if a point is within the defined region."""
        x1, y1, x2, y2 = region
        return x1 <= x <= x2 and y1 <= y <= y2

    if config.verbose:
        print("[Self Operating Computer][operate] Starting operations")

    # Now with smart operations within regions.
    for operation in operations:
        if config.verbose:
            print("[Self Operating Computer][operate] Processing operation", operation)
        
        time.sleep(1)  # Delay for demonstration purposes
        operate_type = operation.get("operation").lower()
        operate_detail = ""
        
        if config.verbose:
            print("[Self Operating Computer][operate] Operation type:", operate_type)

        # Check if operation coordinates are within the defined region, if region is specified
        if region:
            x = operation.get("x", 0)
            y = operation.get("y", 0)
            if not is_within_region(x, y, region):
                print(f"Operation at ({x}, {y}) is outside the defined region and will be skipped.")
                continue  # Skip this operation as it's outside the defined region

        if operate_type == "press" or operate_type == "hotkey":
            keys = operation.get("keys")
            operate_detail = f"keys: {keys}"
            # Press/hotkey operations don’t have coordinates, so they can always run.
            operating_system.press(keys)
        elif operate_type == "write":
            content = operation.get("content")
            operate_detail = f"content: '{content}'"
            
            # Writing might require cursor positioning – ensure it's inside the region.
            x = operation.get("x", 0)  # Assume default position if not specified
            y = operation.get("y", 0)
            
            if not region or is_within_region(x, y, region):
                # Get current cursor position to ensure it starts within the region
                current_x, current_y = pyautogui.position()
                if is_within_region(current_x, current_y, region):
                    operating_system.write(content)
                else:
                    print(f"Write operation aborted because the cursor is outside the region ({current_x}, {current_y}).")
            else:
                print(f"Write action at ({x}, {y}) skipped (out of bounds).")
        elif operate_type == "click":
            x = operation.get("x")
            y = operation.get("y")
            operate_detail = f"click at ({x}, {y})"
            # Check if the click is within the defined region
            if not region or is_within_region(x, y, region):
                operating_system.mouse({"x": x, "y": y})
            else:
                print(f"Click at ({x}, {y}) skipped (out of bounds).")
        elif operate_type == "done":
            summary = operation.get("summary")
            print(f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}] Objective Complete: {ANSI_BLUE}{summary}{ANSI_RESET}\n")
            return True  # Stop operation when done
        else:
            print(f"[{ANSI_GREEN}Self-Operating Computer{ANSI_RED} Error: unknown operation response{ANSI_RESET} {operation}")
            return True  # Stop operation on error

        # Print operation details
        print(f"[{ANSI_GREEN}Self-Operating Computer{ANSI_RESET} | {ANSI_BRIGHT_MAGENTA}{model}{ANSI_RESET}] Thought: {operate_thought}, Action: {ANSI_BLUE}{operate_type}{ANSI_RESET} {operate_detail}\n")

    return False  # Continue if not done

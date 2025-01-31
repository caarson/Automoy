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
    config.validation(model)

    # Initialize region_coords to avoid undefined variable error
    region_coords = None

    ## Boot Arguments:
    if define_region:
        import tkinter as tk
        import threading
        from operate.utils.area_selector import select_area, create_outline_window

        done_event = threading.Event()
        region_coords = []

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

        done_event.wait()
        print(f"Operating within region: {region_coords}")
    
    if voice_mode:
        try:
            from whisper_mic import WhisperMic
            mic = WhisperMic()
        except ImportError:
            print("Voice mode requires the 'whisper_mic' module. Please install it using 'pip install -r requirements-audio.txt'")
            sys.exit(1)
    
    if terminal_prompt:
        objective = terminal_prompt
    elif voice_mode:
        print(f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RESET} Listening for your command... (speak now)")
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Error in capturing voice input: {e}{ANSI_RESET}")
            return
    else:
        print(f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n{USER_QUESTION}")
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    system_prompt = get_system_prompt(model, objective)
    system_message = {"role": "system", "content": system_prompt}
    messages = [system_message]

    loop_count = 0
    session_id = None

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
            print(f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}")
            break
        except Exception as e:
            print(f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}")
            break

def operate(operations, model, region=None):
    if config.verbose:
        print("[Self Operating Computer][operate] Starting operations")

    print(f"[DEBUG] Operations received: {operations}")
    print(f"[DEBUG] Type of operations: {type(operations)}")

    if not operations:
        print(f"{ANSI_RED}[Error] Operations list is empty or None. Exiting operation processing.{ANSI_RESET}")
        return True

    for operation in operations:
        if not isinstance(operation, dict) or "operation" not in operation:
            print(f"{ANSI_RED}[Error] Operation type is missing in {operation}.{ANSI_RESET}")
            continue

        operate_type = operation["operation"].lower()
        operate_detail = ""

        if config.verbose:
            print("[Self Operating Computer][operate] Operation type:", operate_type)

        if operate_type == "press":
            keys = operation.get("keys")
            operate_detail = f"keys: {keys}"
            operating_system.press(keys)
        elif operate_type == "write":
            content = operation.get("text")
            operate_detail = f"content: '{content}'"
            operating_system.write(content)
        elif operate_type == "click":
            text = operation.get("text", "")
            operate_detail = f"click on {text}"
            operating_system.click(text)
        elif operate_type == "done":
            summary = operation.get("summary")
            print(f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}] Objective Complete: {ANSI_BLUE}{summary}{ANSI_RESET}\n")
            return True
        else:
            print(f"[{ANSI_GREEN}Self-Operating Computer{ANSI_RED} Error: unknown operation response{ANSI_RESET} {operation}")
            return True

        print(f"[{ANSI_GREEN}Self-Operating Computer{ANSI_RESET} | {ANSI_BRIGHT_MAGENTA}{model}{ANSI_RESET}] Action: {ANSI_BLUE}{operate_type}{ANSI_RESET} {operate_detail}\n")
    
    return False

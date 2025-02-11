import sys
import os
import time
import asyncio
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
api_source, api_value = config.get_api_source()  # Get preferred API source

def main(model=None, terminal_prompt=None, voice_mode=False, verbose_mode=False, define_region=False):
    """
    Main function for the Automoy.

    It reads from the config.txt (via the Config class) to determine default
    MODEL, DEFINE_REGION, and DEBUG settings, then uses any function arguments
    if provided.
    """

    # Ensure the correct model is chosen based on API source
    if api_source == "lmstudio":
        chosen_model = config.model  # Use LMStudio model from config.txt
    elif api_source == "openai":
        chosen_model = "gpt-4" if not model else model  # Default OpenAI model
    else:
        raise ValueError(f"{ANSI_RED}[Error] No valid API source found!{ANSI_RESET}")

    chosen_define_region = define_region or config.define_region  # from config.txt's DEFINE_REGION
    chosen_verbose_mode = verbose_mode or config.debug  # from config.txt's DEBUG

    from operate.utils.check_cuda import check_cuda
    cuda_avaliable = check_cuda()

    print("CUDA STATUS:" + str(cuda_avaliable))
    if cuda_avaliable:
        print("CUDA enabled, proceeding with startup...")
    else:
        print("CUDA is either not installed or improperly configured or installed.\nExiting...")
        exit()

    config.verbose = chosen_verbose_mode

    # Validate only if using OpenAI
    if api_source == "openai":
        config.validation(chosen_model)

    region_coords = None

    if chosen_define_region:
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
                create_outline_window(region_coords)
            select_area(handle_selection)
            root.mainloop()

        gui_thread = threading.Thread(target=run_gui, daemon=True)
        gui_thread.start()
        done_event.wait()
        print(f"Operating within region: {region_coords}")

    print(f"Using API Source: {api_source}, Endpoint: {api_value}, Model: {chosen_model}")

    loop_count = 0
    session_id = None
    screenshot_path = os.path.join(os.getcwd(), "operate", "data", "screenshots", "screenshot.png")
    os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
    capture_screen_with_cursor(screenshot_path)

    from operate.exceptions import ModelNotRecognizedException

    while True:
        if config.verbose:
            print("[Automoy] loop_count", loop_count)
        try:
            result = asyncio.run(get_next_action(chosen_model, [], terminal_prompt, session_id, screenshot_path))
            if result is None:
                print(f"{ANSI_RED}[Error] get_next_action returned None.{ANSI_RESET}")
                break
            
            operations, session_id = result if isinstance(result, tuple) else ([], None)
            stop = operate(operations, chosen_model, region=region_coords)
            if stop:
                break

            loop_count += 1
            if loop_count > 10:
                break


        except ModelNotRecognizedException as e:
            print(f"{ANSI_GREEN}[Automoy]{ANSI_RED}[Error] -> {e} {ANSI_RESET}")
            break
        except Exception as e:
            print(f"{ANSI_GREEN}[Automoy]{ANSI_RED}[Error] -> {e} {ANSI_RESET}")
            break

def operate(operations, model, region=None):
    if config.verbose:
        print("[Automoy][operate] Starting operations")
    print(f"[DEBUG] Operations received: {operations}")
    if not operations:
        print(f"{ANSI_RED}[Error] Operations list is empty or None. Exiting operation processing.{ANSI_RESET}")
        return True
    
    for operation in operations:
        if not isinstance(operation, dict) or "operation" not in operation:
            print(f"{ANSI_RED}[Error] Operation type is missing in {operation}.{ANSI_RESET}")
            continue

        operate_type = operation["operation"].lower()
        operate_detail = ""
        try:
            if operate_type == "press":
                keys = operation.get("keys")
                operate_detail = f"keys: {keys}"
                operating_system.press(keys)
            elif operate_type == "write":
                content = operation.get("text", "")
                operate_detail = f"content: '{content}'"
                operating_system.write(content)
            elif operate_type == "click":
                text = operation.get("text", "")
                operate_detail = f"click on {text}"
                operating_system.click(text)
            elif operate_type == "done":
                summary = operation.get("summary", "Objective complete.")
                print(f"[{ANSI_GREEN}Automoy {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}] Objective Complete: {ANSI_BLUE}{summary}{ANSI_RESET}\n")
                return True
            else:
                print(f"{ANSI_RED}[Error] Unknown operation type: {operate_type}.{ANSI_RESET}")
                return True

            print(f"[{ANSI_GREEN}Automoy{ANSI_RESET} | {ANSI_BRIGHT_MAGENTA}{model}{ANSI_RESET}] Action: {ANSI_BLUE}{operate_type}{ANSI_RESET} {operate_detail}\n")
        except Exception as e:
            print(f"{ANSI_RED}[Error] Failed to execute operation: {operate_type}, Detail: {operate_detail}, Error: {e}{ANSI_RESET}")
    return False

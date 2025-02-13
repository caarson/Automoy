import sys
import os
import time
import asyncio
import platform

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

# New helper: Ask the AI if it wants a screenshot, including the system prompt.
async def ask_screenshot_preference(chosen_model, objective, session_id, screenshot_path):
    """
    Asks the AI whether it would like a screenshot of the current screen.
    This function builds a payload that includes the full system prompt from get_system_prompt.
    """
    from operate.model_handlers.lmstudio_handler import call_lmstudio_model
    from operate.utils.prompts import get_system_prompt

    # Get the full system prompt (which includes contextual info and the objective)
    system_prompt = get_system_prompt(chosen_model, objective)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Objective: {objective}\nWould you like a screenshot of the current screen? Please reply with 'yes' or 'no'."}
    ]
    # For a quick one-shot answer, we can disable streaming here.
    response = await call_lmstudio_model(messages, "screenshot preference", chosen_model)
    return response

def main(model=None, terminal_prompt=None, voice_mode=False, verbose_mode=False, define_region=False):
    """
    Main function for Automoy.

    Reads configuration settings from Config, then:
      1. If no objective is provided, prompts the user using USER_QUESTION.
      2. Asks the AI if it wants a screenshot of the current screen (using the full system prompt).
      3. If the AI replies "yes", runs OCR/YOLO preprocessing and builds a summary.
         Otherwise, sets a simple "No screenshot provided" summary.
      4. Enters the conversation loop, with follow-up logic if the AI requests more details.
    """
    # Choose the model based on API source.
    if api_source == "lmstudio":
        chosen_model = config.model
    elif api_source == "openai":
        chosen_model = "gpt-4" if not model else model
    else:
        raise ValueError(f"{ANSI_RED}[Error] No valid API source found!{ANSI_RESET}")

    chosen_define_region = define_region or config.define_region
    chosen_verbose_mode = verbose_mode or config.debug

    from operate.utils.check_cuda import check_cuda
    cuda_avaliable = check_cuda()
    print("CUDA STATUS:" + str(cuda_avaliable))
    if cuda_avaliable:
        print("CUDA enabled, proceeding with startup...")
    else:
        print("CUDA is either not installed or improperly configured. Exiting...")
        exit()

    config.verbose = chosen_verbose_mode

    if api_source == "openai":
        config.validation(chosen_model)

    region_coords = None
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

    # If no objective is provided, prompt the user.
    if terminal_prompt is None or terminal_prompt.strip() == "":
        print(USER_QUESTION)
        user_input = input("Objective: ").strip()
        if not user_input:
            print(f"{ANSI_RED}[Error] No objective provided. Exiting...{ANSI_RESET}")
            return
        terminal_prompt = user_input

    # Ask the AI if it wants a screenshot.
    print("[INFO] Asking AI if it wants a screenshot of the current screen...")
    pref_response = asyncio.run(ask_screenshot_preference(chosen_model, terminal_prompt, session_id, screenshot_path))
    print(f"[DEBUG] Screenshot preference response: {pref_response}")
    if "yes" in pref_response.lower():
        from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
        summary_string, full_data = asyncio.run(preprocess_with_ocr_and_yolo(screenshot_path))
    else:
        summary_string = "No screenshot provided."
        full_data = {}

    # Main conversation loop.
    while True:
        if config.verbose:
            print("[Automoy] loop_count", loop_count)
        try:
            messages = [{"role": "system", "content": summary_string}]
            result = asyncio.run(get_next_action(chosen_model, messages, terminal_prompt, session_id, screenshot_path))
            if result is None:
                print(f"{ANSI_RED}[Error] get_next_action returned None.{ANSI_RESET}")
                break

            if isinstance(result, tuple) and len(result) == 3:
                response, session_id, _ = result  # full_data already stored
            else:
                response, session_id = result, None

            print(f"[DEBUG] Initial model response: {response}")

            # Check for trigger phrases in the AI response.
            lower_response = response.lower()
            follow_up_message = None
            if "more ocr" in lower_response or "more details" in lower_response:
                if full_data:
                    extra_ocr = ", ".join(full_data.get("ocr_results", []))
                    follow_up_message = f"Additional OCR data: {extra_ocr}"
                    print("[INFO] Attempting to provide extra OCR details...")
                else:
                    follow_up_message = "No OCR data available."
            elif "more yolo" in lower_response:
                if full_data:
                    extra_yolo = ", ".join(full_data.get("yolo_results", []))
                    follow_up_message = f"Additional YOLO data: {extra_yolo}"
                    print("[INFO] Attempting to provide extra YOLO details...")
                else:
                    follow_up_message = "No YOLO data available."

            if follow_up_message:
                follow_up_messages = [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": follow_up_message}
                ]
                follow_up_result = asyncio.run(get_next_action(chosen_model, follow_up_messages, "Follow-up with extra data", session_id, screenshot_path))
                if isinstance(follow_up_result, tuple) and len(follow_up_result) >= 1:
                    follow_up_response = follow_up_result[0]
                else:
                    follow_up_response = follow_up_result
                print(f"[DEBUG] Follow-up model response: {follow_up_response}")
                response = response + "\n" + follow_up_response

            # Process operations.
            stop = operate(response, chosen_model, region=region_coords)
            if stop:
                break

            loop_count += 1
            if loop_count > 10:
                break

        except ModelNotRecognizedException as e:
            print(f"{ANSI_GREEN}[Automoy]{ANSI_RED}[Error] -> {e}{ANSI_RESET}")
            break
        except Exception as e:
            print(f"{ANSI_GREEN}[Automoy]{ANSI_RED}[Error] -> {e}{ANSI_RESET}")
            break

def operate(operations, model, region=None):
    """
    Processes the model's response. In this example, we treat the response as free-form text.
    If it contains "done", the task is considered complete.
    """
    if config.verbose:
        print("[Automoy][operate] Starting operations")
    print(f"[DEBUG] Operations received: {operations}")
    if not operations:
        print(f"{ANSI_RED}[Error] No operations found. Exiting operation processing.{ANSI_RESET}")
        return True

    if "done" in operations.lower():
        print(f"[{ANSI_GREEN}Automoy{ANSI_RESET} | {ANSI_BRIGHT_MAGENTA}{model}{ANSI_RESET}] Operation complete.")
        return True

    print(f"[{ANSI_GREEN}Automoy{ANSI_RESET} | {ANSI_BRIGHT_MAGENTA}{model}{ANSI_RESET}] Operations: {operations}")
    return False

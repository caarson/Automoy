import sys
import os
import time
import asyncio
import platform
import json

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

# NEW IMPORT: the smart parser
from operate.utils.llm_smart_json_parser import parse_smart_json

# Load configuration
config = Config()
operating_system = OperatingSystem()
api_source, api_value = config.get_api_source()  # Get preferred API source

# Helper: Ask the AI if it wants a screenshot using the system prompt.
async def ask_screenshot_preference(chosen_model, objective, session_id, screenshot_path):
    """
    Asks the AI whether it wants a screenshot of the current screen.

    We remove the extra system role ("screenshot preference") and directly feed the system prompt.
    The user message is constructed from user_message_objective so it matches the new objective.

    The AI can respond with a JSON array, e.g.:
        [{"operation": "take_screenshot", "reason": "Need to see what's on the screen"}]
    """
    from operate.model_handlers.lmstudio_handler import call_lmstudio_model
    system_prompt = get_system_prompt(chosen_model, objective)

    # Construct the user message from the objective.
    user_message_objective = (
        f"Objective: {objective}. "
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_objective}
    ]

    response = await call_lmstudio_model(messages, "screenshot preference", chosen_model)
    return response


def main(model=None, terminal_prompt=None, voice_mode=False, verbose_mode=False, define_region=False):
    """
    Main function for Automoy.

    Steps:
      1. If no objective is provided, prompt the user using USER_QUESTION.
      2. Ask the AI if it wants a screenshot.
         - If the AI replies with a JSON action {"operation": "take_screenshot"}, run OCR/YOLO.
         - Otherwise, skip screenshot processing.
      3. Build the conversation payload with two messages:
           - A system message containing the robust prompt from prompts.py.
           - A user message containing "Objective: <objective>".
      4. Enter the conversation loop and process follow-up requests.
    """
    # Choose the model.
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

    # Prompt for an objective if none is provided.
    if terminal_prompt is None or terminal_prompt.strip() == "":
        print(USER_QUESTION)
        user_input = input("Objective: ").strip()
        if not user_input:
            print(f"{ANSI_RED}[Error] No objective provided. Exiting...{ANSI_RESET}")
            return
        terminal_prompt = user_input

    print("[INFO] Asking AI if it wants a screenshot of the current screen (JSON-based)...")
    pref_response = asyncio.run(ask_screenshot_preference(chosen_model, terminal_prompt, session_id, screenshot_path))
    print(f"[DEBUG] Screenshot preference response (raw): {pref_response}")

    # Use the "parse_smart_json" function from llm_smart_json_parser to handle code fences or extra text.
    parsed_actions = parse_smart_json(pref_response)
    if parsed_actions and isinstance(parsed_actions, list):
        take_screenshot_requested = any(
            (action.get("operation") == "take_screenshot")
            for action in parsed_actions
        )
        if take_screenshot_requested:
            from operate.utils.preprocessing import preprocess_with_ocr_and_yolo
            summary_string, full_data = asyncio.run(preprocess_with_ocr_and_yolo(screenshot_path))
        else:
            summary_string = "No screenshot provided."
            full_data = {}
    else:
        print("[DEBUG] The AI response is not valid JSON or is not a list; no screenshot taken.")
        summary_string = "No screenshot provided."
        full_data = {}

    # Build the conversation payload with two messages:
    system_message_prompt = get_system_prompt(chosen_model, terminal_prompt)
    user_message_objective = f"Objective: {terminal_prompt}"

    conversation_messages = [
        {"role": "system", "content": system_message_prompt},
        {"role": "user", "content": user_message_objective}
    ]

    # Main conversation loop.
    while True:
        if config.verbose:
            print("[Automoy] loop_count", loop_count)
        try:
            result = asyncio.run(
                get_next_action(
                    chosen_model,
                    conversation_messages,
                    terminal_prompt,
                    session_id,
                    screenshot_path
                )
            )

            if result is None:
                print(f"{ANSI_RED}[Error] get_next_action returned None.{ANSI_RESET}")
                break

            if isinstance(result, tuple) and len(result) == 3:
                response, session_id, _ = result  # full_data references are stored
            else:
                response, session_id = result, None

            print(f"[DEBUG] Initial model response: {response}")

            # Check for trigger phrases for additional details.
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

            # If a follow-up message is needed, send it.
            if follow_up_message:
                follow_up_messages = [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": follow_up_message}
                ]
                follow_up_result = asyncio.run(
                    get_next_action(
                        chosen_model,
                        follow_up_messages,
                        "Follow-up with extra data",
                        session_id,
                        screenshot_path
                    )
                )

                if isinstance(follow_up_result, tuple) and len(follow_up_result) >= 1:
                    follow_up_response = follow_up_result[0]
                else:
                    follow_up_response = follow_up_result

                print(f"[DEBUG] Follow-up model response: {follow_up_response}")
                response = response + "\n" + follow_up_response

            # Process the final response.
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
    Processes the model's response. The response is treated as free-form text.
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

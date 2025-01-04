import sys
import os
import time
import asyncio
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt
from operate.exceptions import ModelNotRecognizedException
import platform

# from operate.models.prompts import USER_QUESTION, get_system_prompt
from operate.models.prompts import (
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
from operate.models.apis import get_next_action

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

    mic = None
    # Initialize `WhisperMic`, if `voice_mode` is True

    config.verbose = verbose_mode
    config.validation(model, voice_mode)

    if define_region:
        print("Region definition mode activated.")
        from utils.area_selector import select_area

        region = select_area()
        print(f"Selected region: {region}")
        
        # Continue with the main functionality after defining the region

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

    # Skip message dialog if prompt was given directly
    if not terminal_prompt:
        message_dialog(
            title="Self-Operating Computer",
            text="An experimental framework to enable multimodal models to operate computers",
            style=style,
        ).run()

    else:
        print("Running direct prompt...")

    # # Clear the console
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

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

    while True:
        if config.verbose:
            print("[Self Operating Computer] loop_count", loop_count)
        try:
            operations, session_id = asyncio.run(
                get_next_action(model, messages, objective, session_id)
            )

            stop = operate(operations, model)
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
            operating_system.press(keys)
        elif operate_type == "write":
            content = operation.get("content")
            operate_detail = f"content: '{content}'"
            operating_system.write(content)
        elif operate_type == "click":
            x = operation.get("x")
            y = operation.get("y")
            operate_detail = f"click at ({x}, {y})"
            operating_system.mouse(x, y)
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

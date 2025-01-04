import sys
import platform
import os
from prompt_toolkit.styles import Style as PromptStyle


# Define style
style = PromptStyle.from_dict(
    {
        "dialog": "bg:#cccccc",  # Light grey background
        "button": "bg:#ffffff #000000",  # White background, black text
        "dialog.body": "bg:#eeeeee #000000",  # Very light grey background, black text
        "dialog shadow": "bg:#aaaaaa",  # Grey shadow
    }
)



# Check if on a windows terminal that supports ANSI escape codes
def supports_ansi():
    """
    Check if the terminal supports ANSI escape codes
    """
    plat = platform.system()
    supported_platform = plat != "Windows" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty


# Define ANSI color codes
ANSI_GREEN = "\033[32m" if supports_ansi() else ""  # Standard green text
ANSI_BRIGHT_GREEN = "\033[92m" if supports_ansi() else ""  # Bright/bold green text
ANSI_RESET = "\033[0m" if supports_ansi() else ""  # Reset to default text color
ANSI_BLUE = "\033[94m" if supports_ansi() else ""  # Bright blue
ANSI_YELLOW = "\033[33m" if supports_ansi() else ""  # Standard yellow text
ANSI_RED = "\033[31m" if supports_ansi() else ""
ANSI_BRIGHT_MAGENTA = "\033[95m" if supports_ansi() else ""  # Bright magenta text

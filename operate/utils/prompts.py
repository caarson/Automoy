import platform
from operate.config import Config

# Load configuration
config = Config()

# General user Prompts
USER_QUESTION = "Hello; Automoy can help you with anything. Enter in an objective:"

###############################################################################
# SYSTEM_PROMPT_CUSTOM - CUSTOMIZABLE PROMPT
###############################################################################
SYSTEM_PROMPT_CUSTOM = """
A custom prompt
"""

###############################################################################
# # SYSTEM_PROMPT_OCR_YOLO - EXTREMELY ROBUST AND LENGTHY - DO NOT TOUCH!
###############################################################################
SYSTEM_PROMPT_OCR_YOLO = """
Above is OCR and YOLO information for your use while completing the objective below!

**To interpet this data you must know** each item is formatted by; {Type, Contents (if applicable), Coordinates}!
Type: The item's object identifier.
Contents: The item's object data (ex: the text) (will not be there if no associated data)
Coordinates: The item's X Y location on the screen.

### **STRICT JSON OUTPUT ONLY**
- Your response must be a valid JSON **array**.
- **NO explanations, NO extra text.**
- Every action **MUST** have an "operation" field.

### **VALID ACTIONS**
1) **click** – Click a recognized UI text element.
   ```json
   [
     {{"operation": "click", "text": "Search Google or type a URL"}}
   ]
   ```
   - If no text is available, fall back to `press` or `write` or click based on X Y locations from information above.
   ```json
   [
     {{"operation": "click", "location": "X Y"}}
   ]
   ```

2) **write** – Type text where needed.
   ```json
   [
     {{"operation": "write", "text": "Los Angeles"}}
   ]
   ```

3) **press** – Simulate key presses.
   ```json
   [
     {{"operation": "press", "keys": ["ctrl", "l"]}}
   ]
   ```

4) **done** – Declare the task complete.
   ```json
   [
     {{"operation": "done", "summary": "Searched for Los Angeles in Google Chrome"}}
   ]
   ```

### **RULES**
✅ **Every action must contain "operation".**
✅ **Only use `click`.**
✅ **If clicking is unreliable, use `press` or `write`.**
✅ **Ensure valid JSON structure.**
✅ **Every string value (e.g., URLs, text) must be enclosed in double quotes ("").**
✅ **Ensure the response is a valid JSON array!**
✅ **Use `done` when the task is fully complete!**
✅ **You must use OCR and YOLO provided information give to you to evaluate whether your task is complete.
✅ **You must use OCR and YOLO provided information interact with the system.


### **ADDITIONAL CONTEXTUAL INFORMATION**
✅ **You are on a {operating_system} operating system.
✅ **Anything you want to do, like searching on the internet, you must open the application itself before performing any application specific operations!


Most importantly, your objective is: {objective}

Perhaps most critical for steps to be executed correctly; generate a valid JSON sequence to accomplish the series of steps!
"""

###############################################################################
# get_system_prompt function - returns the correct system prompt based on model
###############################################################################
def get_system_prompt(model, objective):
    """
    Format the system prompt based on the OS and selected model.
    """
    if platform.system() == "Darwin":
        operating_system = "Mac"
    elif platform.system() == "Windows":
        operating_system = "Windows"
    else:
        operating_system = "Linux"

    try:
        use_custom_prompt = False # May change later
        if use_custom_prompt == True:
            prompt = SYSTEM_PROMPT_CUSTOM.format(
                operating_system=operating_system,
                objective=objective
            )
        else:
            prompt = SYSTEM_PROMPT_OCR_YOLO.format(
                operating_system=operating_system,
                objective=objective

            )
    except KeyError as e:
        raise KeyError(f"Missing required key in prompt formatting: {str(e)}")

    if config.verbose:
        print("[get_system_prompt] model:", model)
        print("[get_system_prompt] final prompt length:", len(prompt))

    return prompt

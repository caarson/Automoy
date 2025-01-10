import os
import time
import base64
import traceback
import json
from openai import OpenAI
from PIL import Image
from operate.config import Config
from operate.utils.screenshot import capture_screen_with_cursor
from operate.utils.prompts import get_user_prompt, get_user_first_message_prompt

# Load configuration
config = Config()

class OpenAIHandler:
    def __init__(self):
        self.client = config.initialize_openai()

    def call_gpt_4o(self, messages):
        """
        Sends a request to GPT-4 model and returns the response.
        """
        if config.verbose:
            print("[OpenAIHandler][call_gpt_4o] Sending request to OpenAI...")

        try:
            screenshot_filename = "screenshots/screenshot.png"
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")

            # Capture the screen
            capture_screen_with_cursor(screenshot_filename)

            with open(screenshot_filename, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            user_prompt = (
                get_user_first_message_prompt()
                if len(messages) == 1
                else get_user_prompt()
            )

            vision_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
            messages.append(vision_message)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                presence_penalty=1,
                frequency_penalty=1,
            )

            content = response.choices[0].message.content
            assistant_message = {"role": "assistant", "content": json.loads(content)}
            messages.append(assistant_message)

            return json.loads(content)

        except Exception as e:
            print("[OpenAIHandler][Error] Exception occurred:", e)
            traceback.print_exc()
            return self.call_gpt_4o(messages)

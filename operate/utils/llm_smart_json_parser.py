import json
import re

def extract_json_from_fence(response_text: str) -> str:
    """
    Searches for a triple-backtick code fence labeled as json (```json ... ```).
    If found, returns the substring within those fences. Otherwise, returns None.
    """
    pattern = r"```json(.*?)```"  # Regex to find code blocks labeled 'json'
    match = re.search(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_json_strict_brackets(response_text: str) -> str:
    """
    Attempts to extract the first '[' and the last ']' from the response,
    returning the substring. This helps if there's extra text around the JSON.
    """
    start = response_text.find("[")
    end = response_text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return response_text[start:end+1]


def parse_smart_json(response_text: str):
    """
    1. Attempt to locate a ```json ...``` fenced block. If found,
       parse that as JSON.
    2. Otherwise, find the first '[' and last ']' in the text
       and parse that substring.
    3. If no valid JSON can be parsed, return None.
    """
    # 1) Check for code fence
    fenced = extract_json_from_fence(response_text)
    if fenced:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass  # fallback to bracket approach

    # 2) If no fence or parse fail, try bracket approach
    bracketed = extract_json_strict_brackets(response_text)
    if bracketed:
        try:
            return json.loads(bracketed)
        except json.JSONDecodeError:
            pass

    # 3) Return None if nothing worked
    return None

import os

class Config:
    """
    Configuration class for managing settings from a config.txt file.

    By default, this expects config.txt to be in the same directory as this script.
    Adjust paths as needed.

    Expected config.txt contents:
    ####
    # APIs
    ####

    OPENAI_API_KEY:
    <key goes here>

    LMSTUDIO_API_URL:
    <http://127.0.0.1:1234>

    ####
    # OPERATE PARAMETERS
    ####

    MODEL:
    <deepseek-r1-distill-qwen-7b>

    DEFINE_REGION:
    <True>

    DEBUG:
    <True>

    Usage:
      config = Config()      # will read config.txt from the same directory
      config.validation(...) # optional: run checks on your chosen model
    """

    _instance = None

    def __new__(cls):
        # Implement a singleton pattern (only one Config object).
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        # Prevent re-initializing if we've already done so.
        if getattr(self, "_initialized", False):
            return

        # If config_path is None, assume it's in the same directory as this file.
        if config_path is None:
            current_dir = os.path.dirname(__file__)
            self.config_path = os.path.join(current_dir, 'config.txt')
        else:
            self.config_path = config_path

        self._initialized = False

        # Default attributes (will be overwritten if present in config.txt).
        self.openai_api_key = None
        self.lmstudio_api_url = None
        self.model = None
        self.define_region = False
        self.debug = False

        # Parse the config.txt file to populate these attributes
        self.load_config()
        self._initialized = True

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config.txt not found at: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            # Strip whitespace
            line = line.strip()

            # Skip comments or empty lines
            if not line or line.startswith("#"):
                continue

            # Check if it's a key-value pair of the form KEY:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                print(f"DEBUG: Read {key} -> {value}")

                if key.upper() == "OPENAI_API_KEY":
                    self.openai_api_key = value.strip("<>") or None
                elif key.upper() == "LMSTUDIO_API_URL":
                    self.lmstudio_api_url = value.strip("<>") or None
                elif key.upper() == "MODEL":
                    self.model = value.strip("<>")
                elif key.upper() == "DEFINE_REGION":
                    self.define_region = value.strip("<>").lower() == "true"
                elif key.upper() == "DEBUG":
                    self.debug = value.strip("<>").lower() == "true"

    def get_api_source(self):
        """
        Returns the available API source, preferring OpenAI if both are available.
        """
        if self.openai_api_key:
            return "openai", self.openai_api_key
        elif self.lmstudio_api_url:
            return "lmstudio", self.lmstudio_api_url
        else:
            raise ValueError("Both OpenAI API Key and LMStudio URL are missing. At least one must be provided.")

    def validation(self, model: str):
        print(f"DEBUG: Validating model -> {model}")
        if model.lower().startswith("gpt") or model.lower().startswith("openai"):
            if not self.openai_api_key:
                raise ValueError(
                    "[Config][validation] Missing required OPENAI_API_KEY for an OpenAI-based model, fallback to LMStudio attempted."
                )

if __name__ == '__main__':
    # Create a config object (reads config.txt automatically)
    config = Config()
    
    # Optionally validate against your current model choice
    model = config.model or "openai-gpt"
    config.validation(model)

    api_source, api_value = config.get_api_source()
    print(f"Using API Source: {api_source}")
    print(f"API Key/URL: {api_value}")
    print("MODEL:", config.model)
    print("DEFINE_REGION:", config.define_region)
    print("DEBUG:", config.debug)

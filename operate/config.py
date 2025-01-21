import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from prompt_toolkit.shortcuts import input_dialog
import requests  # For testing Ollama API
import subprocess


class Config:
    """
    Configuration class for managing settings.

    Attributes:
        verbose (bool): Flag indicating whether verbose mode is enabled.
        openai_api_key (str): API key for OpenAI.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        load_dotenv()  # Load environment variables from .env
        self.verbose = False
        self.openai_api_key = None  # Backup if .env is not set
        self.deepseek_api_key = None  # Placeholder for DeepSeek if needed
        self.ollama_url = "http://localhost:11434/api/generate"  # Ollama's default local API URL

    # ---------------------
    # OpenAI Initialization
    # ---------------------
    def initialize_openai(self):
        """
        Initializes OpenAI client with API key.
        """
        if self.verbose:
            print("[Config][initialize_openai]")

        if self.openai_api_key:
            if self.verbose:
                print("[Config][initialize_openai] using cached openai_api_key")
            api_key = self.openai_api_key
        else:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            self.prompt_and_save_api_key("OPENAI_API_KEY", "OpenAI API key")
            api_key = self.openai_api_key

        client = OpenAI(api_key=api_key)
        client.api_key = api_key
        return client

    # ---------------------
    # Ollama Initialization
    # ---------------------
    def initialize_ollama(self):
        """
        Initializes Ollama by checking if the local API is running.
        """
        if self.verbose:
            print("[Config][initialize_ollama] Checking Ollama server...")

        try:
            response = requests.get(self.ollama_url.replace("/generate", "/info"))
            if response.status_code == 200:
                if self.verbose:
                    print("[Config][initialize_ollama] Ollama server is running.")
                return True
            else:
                print("[Config][initialize_ollama] Ollama server did not respond.")
                return False
        except requests.ConnectionError:
            print("[Config][initialize_ollama] Ollama is not running. Attempting to start Ollama...")
            self.start_ollama_server()

    def start_ollama_server(self):
        """
        Starts the Ollama local API server if it is not running.
        """
        try:
            subprocess.Popen(["ollama", "serve"])
            print("[Config][start_ollama_server] Ollama server started.")
        except FileNotFoundError:
            sys.exit("Ollama CLI not found. Please install Ollama: https://ollama.com")

    # ---------------------
    # DeepSeek Initialization (if required)
    # ---------------------
    def initialize_deepseek(self):
        """
        Placeholder for DeepSeek API key or any initialization steps.
        """
        if self.verbose:
            print("[Config][initialize_deepseek] Checking DeepSeek API setup.")

        # Optionally require an API key for DeepSeek if needed
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            print("[Config] DeepSeek does not require an API key or it’s not set.")
            # If DeepSeek requires a server to run, add server initialization here.

    # ---------------------
    # API Key Management
    # ---------------------
    def validation(self, model):
        """
        Validate the input parameters for the dialog operation, ensuring API key presence.
        """
        if model.startswith("gpt") or model.startswith("openai"):
            self.require_api_key("OPENAI_API_KEY", "OpenAI API key", True)
        elif model == "deepseek":
            print("[Config] No DeepSeek API key validation needed.")
        elif model.startswith("ollama"):
            self.initialize_ollama()

    def require_api_key(self, key_name, key_description, is_required):
        key_exists = bool(os.environ.get(key_name))
        if self.verbose:
            print(f"[Config] require_api_key for {key_name}: {key_exists}")
        if is_required and not key_exists:
            self.prompt_and_save_api_key(key_name, key_description)

    def prompt_and_save_api_key(self, key_name, key_description):
        key_value = input_dialog(
            title="API Key Required", text=f"Please enter your {key_description}:"
        ).run()

        if key_value is None:
            sys.exit("Operation cancelled by user.")

        if key_value:
            if key_name == "OPENAI_API_KEY":
                self.openai_api_key = key_value
            elif key_name == "DEEPSEEK_API_KEY":
                self.deepseek_api_key = key_value
            self.save_api_key_to_env(key_name, key_value)
            load_dotenv()  # Reload environment variables

    @staticmethod
    def save_api_key_to_env(key_name, key_value):
        with open(".env", "a") as file:
            file.write(f"\n{key_name}='{key_value}'")

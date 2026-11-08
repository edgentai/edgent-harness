import os
import time
from pathlib import Path
from typing import List, Dict, Any, Callable
from dotenv import load_dotenv

# Automatically load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

from edgent.tools.file_ops import read_file, write_file, list_dir
from edgent.tools.cmd_ops import run_cmd
from edgent.tools.search_ops import search_files

AVAILABLE_TOOLS: Dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_cmd": run_cmd,
    "search_files": search_files,
}

# Verified model strings for the google-genai SDK
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]

class AgentLoop:
    def __init__(self, model_name: str = "gemini-3.8-flash", max_iterations: int = 5):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_iterations = max_iterations
        self._init_chat(self.model_name)

    def _init_chat(self, model_name: str):
        """Initializes a persistent chat session using a validated model string."""
        self.model_name = model_name
        system_instruction = (
            "You are Edgent, an autonomous agent harness. "
            "Use workspace tools to inspect files, execute commands, search codebases, and modify files safely. "
            "Always inspect directory structure or search files before making assumptions."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[read_file, write_file, list_dir, run_cmd, search_files],
            temperature=0.2,
        )

        self.chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )

    def _send_with_retry_and_fallback(self, message, max_retries: int = 3):
        """Retries API calls on 503 errors and automatically falls back to secondary models on 429 quota/404 model errors."""
        current_model_idx = 0
        if self.model_name in FALLBACK_MODELS:
            current_model_idx = FALLBACK_MODELS.index(self.model_name)
        else:
            FALLBACK_MODELS.insert(0, self.model_name)

        while current_model_idx < len(FALLBACK_MODELS):
            active_model = FALLBACK_MODELS[current_model_idx]
            if self.model_name != active_model:
                print(f"\n[Model Switch]: Changing active model to '{active_model}'...")
                try:
                    self._init_chat(active_model)
                except Exception as init_err:
                    print(f"[Init Error on '{active_model}']: {init_err}")
                    current_model_idx += 1
                    continue

            for attempt in range(max_retries):
                try:
                    return self.chat.send_message(message)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                        print(f"\n[Model Error on '{active_model}']: {e}")
                        print(f"--> Falling back to next model in pool...")
                        break
                    
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = (attempt + 1) * 3
                    print(f"\n[API Spike]: Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)

            current_model_idx += 1

        raise RuntimeError("All available fallback models encountered errors or exhausted rate limits.")

    def run(self, user_prompt: str) -> str:
        print(f"\n[User Prompt]: {user_prompt}")
        response = self._send_with_retry_and_fallback(user_prompt)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            function_calls = response.function_calls
            if not function_calls:
                return response.text or "Completed without text output."

            tool_responses = []
            for call in function_calls:
                tool_name = call.name
                tool_args = call.args or {}
                print(f"[Executing Tool]: {tool_name}({tool_args})")

                try:
                    if tool_name in AVAILABLE_TOOLS:
                        result = str(AVAILABLE_TOOLS[tool_name](**tool_args))
                    else:
                        result = f"Error: Tool '{tool_name}' not recognized."
                except Exception as err:
                    result = f"Tool Execution Exception [{tool_name}]: {type(err).__name__} - {err}"

                if len(result) > MAX_TOOL_OUTPUT_CHARS:
                    result = result[:MAX_TOOL_OUTPUT_CHARS] + f"\n... [Truncated: {len(result) - MAX_TOOL_OUTPUT_CHARS} characters omitted]"

                print(f"[Tool Output Preview]: {result[:120]}..." if len(result) > 120 else f"[Tool Output]: {result}")

                tool_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

            response = self._send_with_retry_and_fallback(tool_responses)

        return "Loop stopped: Max iteration depth reached."
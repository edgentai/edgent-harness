import os
from typing import List, Dict, Any, Callable
import time

from dotenv import load_dotenv
# Automatically load environment variables from .env file
load_dotenv(override=True)

from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from edgent.tools.file_ops import read_file, write_file, list_dir
from edgent.tools.cmd_ops import run_cmd

AVAILABLE_TOOLS: Dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_cmd": run_cmd,
}
#gemini-3.8-flash
class AgentLoop:
    def __init__(self, model_name: str = "gemini-3.6-flash", max_iterations: int = 5):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        
        self.client = genai.Client()
        self.model_name = model_name
        self.max_iterations = max_iterations

    def _send_with_retry(self, chat, message, max_retries: int = 3):
        """Helper to retry API calls when 503/server errors occur."""
        for attempt in range(max_retries):
            try:
                return chat.send_message(message)
            except (ServerError, APIError) as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"\n[API Spike Detected]: Retrying in 2 seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(2)

    def run(self, user_prompt: str) -> str:
        print(f"\n[User Request]: {user_prompt}")
        
        system_instruction = "You are Edgent, an autonomous agent harness. Use your tools to read, list, and write workspace files."

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[read_file, write_file, list_dir,run_cmd],
            temperature=0.2,
        )

        # Create a chat session with automatic execution disabled
        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )

        response = self._send_with_retry(chat, user_prompt)
        
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
                tool_args = call.args
                print(f"[Executing Tool Call]: {tool_name}({tool_args})")

                if tool_name in AVAILABLE_TOOLS:
                    result = AVAILABLE_TOOLS[tool_name](**tool_args)
                else:
                    result = f"Error: Tool '{tool_name}' not recognized."

                print(f"[Tool Result]: {result}")
                
                tool_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

            # Send the tool output back to the model chat session
            response = self._send_with_retry(chat, tool_responses)


        return "Loop stopped: Max iterations reached."
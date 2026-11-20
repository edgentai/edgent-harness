import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env file relative to project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)

print("Available models for your API key:\n" + "-" * 40)
for model in client.models.list():
    # Filter for content generation models
    if hasattr(model, 'supported_actions') and "generateContent" in (model.supported_actions or []):
        print(f"Name: {model.name}")
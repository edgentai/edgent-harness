import os

def read_file(path: str) -> str:
    """Reads text from a file so the AI can see what is inside."""
    if not os.path.exists(path):
        return f"Error: File '{path}' does not exist."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def write_file(path: str, content: str) -> str:
    """Writes text to a file so the AI can create or edit code."""
    try:
        # Create folder if it doesn't exist yet
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Wrote to '{path}'."
    except Exception as e:
        return f"Error writing to file '{path}': {str(e)}"

def list_dir(path: str = ".") -> str:
    """Lists all files in a folder so the AI knows what files exist."""
    if not os.path.exists(path):
        return f"Error: Directory '{path}' does not exist."
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "Folder is completely empty."
    except Exception as e:
        return f"Error listing folder '{path}': {str(e)}"
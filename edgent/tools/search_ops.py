import os
import re

def search_files(query: str, root_dir: str = ".", file_pattern: str = None) -> str:
    """Searches for a regex pattern or keyword across files in the workspace.
    
    Args:
        query: Pattern or keyword to search for.
        root_dir: Workspace directory to start search (default '.').
        file_pattern: Optional file extension filter (e.g., '.py').
    """
    matches = []
    ignore_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "edgent_harness.egg-info"}

    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid search pattern - {e}"

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if file_pattern and not file.endswith(file_pattern):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{line_num}: {line.strip()}")
            except Exception:
                continue

    if not matches:
        return f"No matches found for '{query}'."

    if len(matches) > 50:
        return "\n".join(matches[:50]) + f"\n... [{len(matches) - 50} more matches omitted]"

    return "\n".join(matches)
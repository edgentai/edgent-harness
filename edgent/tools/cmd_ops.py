import subprocess
import os
from typing import Dict, Any

# Safety block: list commands that Edgent is strictly prohibited from running
DISALLOWED_COMMANDS = [
    "rmdir /s", "del /f", "format", "shutdown", "reboot"
]

def run_cmd(command: str, cwd: str = ".") -> str:
    """Executes a shell command in the specified directory and returns the combined stdout/stderr output.
    
    Args:
        command: The shell command to execute (e.g., 'python -m pytest', 'git status').
        cwd: The directory to run the command in (defaults to current working directory).
    """
    # Safety check
    cmd_lower = command.lower()
    for forbidden in DISALLOWED_COMMANDS:
        if forbidden in cmd_lower:
            return f"Error: Command '{command}' blocked for safety reasons."

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30  # Safety timeout (30 seconds)
        )
        
        output = []
        if result.stdout:
            output.append(f"[stdout]:\n{result.stdout.strip()}")
        if result.stderr:
            output.append(f"[stderr]:\n{result.stderr.strip()}")
            
        if not output:
            return f"Command executed with return code {result.returncode} (No output)."
            
        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"
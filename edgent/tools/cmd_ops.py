import subprocess
import shlex

PROHIBITED_COMMANDS = ["rm -rf /", "mkfs", "dd", "format", "shutdown", "reboot"]

def run_cmd(command: str, timeout: int = 30) -> str:
    """Executes a shell command safely with execution timeouts.
    
    Args:
        command: The terminal command to execute.
        timeout: Maximum execution time in seconds (default 30s).
    """
    for forbidden in PROHIBITED_COMMANDS:
        if forbidden in command.lower():
            return f"Error: Command '{command}' blocked by safety guardrails."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        
        return output.strip() if output.strip() else "Command executed with no output."
    
    except subprocess.TimeoutExpired:
        return f"Error: Command '{command}' timed out after {timeout} seconds."
    except Exception as e:
        return f"Execution Error: {e}"
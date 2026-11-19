"""Tools for Coder-Tester Agent: code fence stripping and subprocess code execution."""

import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Any


def extract_all_code_blocks(text: str) -> List[str]:
    """Extract all code blocks enclosed in markdown triple-backtick fences.
    
    Supports ```python ... ``` and generic ``` ... ``` blocks.
    """
    if not text:
        return []
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return []


def strip_code_fences(text: str) -> str:
    """Return the code inside ```python ... ``` or the first fenced block.
    
    If no closing fence is found or no fences exist, returns the stripped text.
    """
    if not text:
        return ""
    blocks = extract_all_code_blocks(text)
    if blocks:
        return blocks[0]
    
    # Handle single opening fence without closing fence
    match = re.search(r"```(?:python)?\s*\n?(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if match and "```" in text:
        content = match.group(1).replace("```", "")
        return content.strip()
        
    return text.strip()


def run_python(code: str, timeout: int = 5) -> Dict[str, Any]:
    """Write code to a temporary file, run with subprocess in the same interpreter.
    
    Returns:
        dict: {"ok": bool, "output": str (last 6 lines of stdout+stderr)}
    Handles TimeoutExpired and execution exceptions.
    """
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    )
    temp_path = temp_file.name

    try:
        temp_file.write(code)
        temp_file.close()

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        combined = (result.stdout or "") + (result.stderr or "")
        lines = [line for line in combined.splitlines() if line.strip()]
        last_lines = "\n".join(lines[-6:]) if lines else ""

        return {
            "ok": result.returncode == 0,
            "output": last_lines
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "output": f"TimeoutExpired: Execution timed out after {timeout} seconds."
        }
    except Exception as e:
        return {
            "ok": False,
            "output": f"Execution error: {e}"
        }
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass

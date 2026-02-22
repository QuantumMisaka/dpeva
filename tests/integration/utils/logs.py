import time
from pathlib import Path
from typing import Optional


def wait_for_text_in_file(
    file_path: Path,
    needle: str,
    timeout_s: float,
    poll_s: float = 5.0,
    min_size: int = 1,
) -> None:
    start = time.time()
    last_err: Optional[Exception] = None
    while True:
        if time.time() - start > timeout_s:
            tail = tail_text(file_path, max_lines=80)
            details = f"\n\nLast exception: {last_err}" if last_err else ""
            raise TimeoutError(
                f"Timeout waiting for '{needle}' in {file_path}\n\nLast lines:\n{tail}{details}"
            )
        try:
            if file_path.exists() and file_path.stat().st_size >= min_size:
                content = file_path.read_text(errors="replace")
                if needle in content:
                    return
        except Exception as e:
            last_err = e
        time.sleep(poll_s)


def tail_text(file_path: Path, max_lines: int = 50) -> str:
    try:
        if not file_path.exists():
            return "<log file not found>"
        lines = file_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"<failed to read log: {e}>"


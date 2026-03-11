"""
ComfyUI-4K4D Subprocess Runner
==============================
Safe async subprocess execution with real-time progress streaming
to ComfyUI's progress bar. All EasyVolcap commands (evc-train, evc-test)
run through this module.

Features:
- Line-by-line stdout streaming with progress parsing
- ComfyUI ProgressBar integration via PromptServer.send_sync()
- Automatic log file writing
- Timeout and cancellation support
- Never raises exceptions to caller — returns SubprocessResult
"""

import os
import re
import subprocess
import threading
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Iterator

logger = logging.getLogger("4K4D.subprocess_runner")


@dataclass
class SubprocessResult:
    """Result of a subprocess execution."""
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    was_cancelled: bool = False
    was_timeout: bool = False
    log_path: str = ""
    error_summary: str = ""

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.was_cancelled and not self.was_timeout


# ─── Progress Parsers ─────────────────────────────────────────────────────────

def evc_train_progress_parser(line: str) -> Optional[tuple]:
    """
    Parse EasyVolcap training output for progress info.

    Matches patterns like:
    - 'iter 1234/1600'
    - 'Iter: 200/1600'
    - 'epoch 5/100'
    - Progress: 50.0%

    Returns:
        (current, total) tuple or None if no match
    """
    # Match iter X/Y pattern
    match = re.search(r'[Ii]ter[:\s]+(\d+)\s*/\s*(\d+)', line)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Match epoch X/Y pattern
    match = re.search(r'[Ee]poch[:\s]+(\d+)\s*/\s*(\d+)', line)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Match percentage
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
    if match:
        pct = float(match.group(1))
        return int(pct), 100

    return None


def evc_psnr_parser(line: str) -> Optional[float]:
    """Parse PSNR value from EasyVolcap output."""
    match = re.search(r'[Pp][Ss][Nn][Rr][:\s]+(\d+\.?\d*)', line)
    if match:
        return float(match.group(1))
    return None


def pip_install_progress_parser(line: str) -> Optional[tuple]:
    """Parse pip install output for progress."""
    # Match: 'Downloading package-1.0.tar.gz (50%)'
    match = re.search(r'(\d+)%', line)
    if match:
        return int(match.group(1)), 100
    return None


def generic_progress_parser(line: str) -> Optional[tuple]:
    """Generic progress parser matching common patterns."""
    # [50/100] or (50/100)
    match = re.search(r'[\[\(](\d+)\s*/\s*(\d+)[\]\)]', line)
    if match:
        return int(match.group(1)), int(match.group(2))

    # 50% or 50.5%
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
    if match:
        return int(float(match.group(1))), 100

    return None


class SubprocessRunner:
    """
    Runs external processes with real-time progress streaming to ComfyUI.

    Usage:
        runner = SubprocessRunner("Train4K4D", "/path/to/logs")
        result = runner.run(
            cmd=["evc-train", "-c", "config.yaml"],
            progress_parser=evc_train_progress_parser,
            unique_id="node_123",
        )
        if result.success:
            print("Training complete!")
        else:
            print(f"Failed: {result.error_summary}")
    """

    def __init__(self, node_name: str, log_dir: str):
        self.node_name = node_name
        self.log_dir = log_dir
        self._cancel_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None

    def run(
        self,
        cmd: list,
        cwd: str = None,
        env: dict = None,
        progress_parser: Callable = None,
        timeout_seconds: int = None,
        unique_id: str = None,
    ) -> SubprocessResult:
        """
        Execute a command with real-time progress streaming.

        Args:
            cmd: Command and arguments as a list
            cwd: Working directory
            env: Environment variables (None = inherit)
            progress_parser: Function(line) -> (current, total) or None
            timeout_seconds: Max execution time (None = no limit)
            unique_id: ComfyUI node unique_id for progress bar updates

        Returns:
            SubprocessResult with return code, output, timing, etc.
        """
        result = SubprocessResult()
        start_time = time.time()

        # Setup log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(self.log_dir) / f"{self.node_name}_{timestamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result.log_path = str(log_path)

        cmd_str = " ".join(str(c) for c in cmd)
        logger.info(f"[{self.node_name}] Running: {cmd_str}")

        try:
            # Merge environment
            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            # Open process
            # stdin=DEVNULL prevents child processes (e.g. apt-get) from
            # reading stdin, which would send SIGTTIN and stop ComfyUI
            # when it runs as a background process.
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=run_env,
            )

            # Stream output
            output_lines = []
            with open(log_path, "w") as log_file:
                log_file.write(f"Command: {cmd_str}\n")
                log_file.write(f"CWD: {cwd or os.getcwd()}\n")
                log_file.write(f"Started: {datetime.now().isoformat()}\n")
                log_file.write("-" * 80 + "\n")

                for line in self._stream_output(
                    self._process, progress_parser, unique_id, timeout_seconds
                ):
                    log_file.write(line + "\n")
                    log_file.flush()
                    output_lines.append(line)

            # Wait for process to complete
            if self._process.poll() is None:
                self._process.wait(timeout=10)

            result.return_code = self._process.returncode
            result.stdout = "\n".join(output_lines)
            result.duration_seconds = time.time() - start_time

            if self._cancel_event.is_set():
                result.was_cancelled = True
                result.error_summary = "Process was cancelled by user."

            if result.return_code != 0 and not result.was_cancelled:
                # Extract last 20 lines as error summary
                last_lines = output_lines[-20:] if len(output_lines) > 20 else output_lines
                result.error_summary = (
                    f"Command failed with return code {result.return_code}.\n"
                    f"Last output:\n" + "\n".join(last_lines)
                )

        except subprocess.TimeoutExpired:
            result.was_timeout = True
            result.error_summary = (
                f"Command timed out after {timeout_seconds} seconds. "
                "This might indicate the process is stuck. Check the log file "
                f"at {log_path} for details."
            )
            if self._process:
                self._process.kill()
                self._process.wait(timeout=5)

        except FileNotFoundError:
            result.return_code = -1
            result.error_summary = (
                f"Command not found: '{cmd[0]}'. "
                "Make sure all dependencies are installed. "
                "Run the DependencyInstall node first."
            )

        except Exception as e:
            result.return_code = -1
            result.error_summary = (
                f"Unexpected error running command: {str(e)}. "
                f"Check log file at {log_path} for details."
            )
            logger.error(f"[{self.node_name}] Subprocess error: {e}", exc_info=True)

        finally:
            self._process = None
            result.duration_seconds = time.time() - start_time
            logger.info(
                f"[{self.node_name}] Completed in {result.duration_seconds:.1f}s "
                f"(rc={result.return_code})"
            )

        return result

    def _stream_output(
        self,
        process: subprocess.Popen,
        progress_parser: Optional[Callable],
        unique_id: Optional[str],
        timeout_seconds: Optional[int],
    ) -> Iterator[str]:
        """Stream process output line by line, yielding each line."""
        start_time = time.time()

        for line in iter(process.stdout.readline, ""):
            if not line:
                break

            line = line.rstrip("\n\r")
            if not line:
                continue

            yield line

            # Parse progress
            if progress_parser and unique_id:
                progress = progress_parser(line)
                if progress is not None:
                    current, total = progress
                    self._send_progress(unique_id, current, total, line.strip())

            # Check cancellation
            if self._cancel_event.is_set():
                logger.info(f"[{self.node_name}] Cancellation requested")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                logger.warning(f"[{self.node_name}] Timeout after {timeout_seconds}s")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

    def _send_progress(self, unique_id: str, current: int, total: int, message: str = "") -> None:
        """Send progress update to ComfyUI's frontend."""
        try:
            # Try ComfyUI's PromptServer
            from server import PromptServer
            PromptServer.instance.send_sync(
                "4k4d.progress",
                {
                    "node": self.node_name,
                    "unique_id": unique_id,
                    "value": current,
                    "max": total,
                    "text": message,
                },
            )
        except Exception:
            # Not running inside ComfyUI, or PromptServer not available
            pass

        try:
            # Also try ComfyUI's built-in progress bar
            import comfy.utils
            if not hasattr(self, '_progress_bar'):
                self._progress_bar = comfy.utils.ProgressBar(total)
            self._progress_bar.update_absolute(current, total)
        except Exception:
            pass

    def cancel(self) -> None:
        """Request cancellation of the running process."""
        self._cancel_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run_simple(self, cmd: list, cwd: str = None, env: dict = None, timeout: int = 300) -> SubprocessResult:
        """
        Run a simple command without progress streaming.
        Convenience wrapper for short-lived commands like pip install.
        """
        return self.run(cmd=cmd, cwd=cwd, env=env, timeout_seconds=timeout)

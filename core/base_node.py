"""
ComfyUI-4K4D Base Node
======================
Abstract base class for all ComfyUI-4K4D nodes.

All 14 nodes inherit from BaseEasyVolcapNode, which provides:
- Standardized error handling (never crash ComfyUI)
- Logging to per-node log files
- Access to EnvManager singleton
- SubprocessRunner factory
- DATASET_INFO validation and immutable updates
- Progress reporting helpers

To add a new method beyond 4K4D (e.g., 3DGS+T, ENeRF), inherit this class
and override METHOD_NAME.
"""

import abc
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .constants import DATASET_INFO_TYPE, CATEGORY_PREFIX
from .env_manager import EnvManager
from .subprocess_runner import SubprocessRunner

logger = logging.getLogger("4K4D.base_node")


class BaseEasyVolcapNode(abc.ABC):
    """
    Abstract base class for all ComfyUI-4K4D nodes.

    Subclasses MUST define:
    - CATEGORY (str): e.g., "4K4D/Setup"
    - FUNCTION (str): e.g., "execute"
    - RETURN_TYPES (tuple): e.g., ("DATASET_INFO", "STRING")
    - RETURN_NAMES (tuple): e.g., ("dataset_info", "status_text")

    Subclasses SHOULD override:
    - _run(): The actual node logic (called by execute via _safe_execute)
    """

    # ── Class Attributes (override in subclasses) ─────────────────────────
    CATEGORY = CATEGORY_PREFIX
    FUNCTION = "execute"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = False
    METHOD_NAME = "4k4d"  # Override for other methods: "3dgs_t", "enerf", etc.

    def __init__(self):
        """Initialize with environment manager and logger."""
        self.env = EnvManager.get_instance()
        self.node_name = self.__class__.__name__
        self._node_logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Create a file logger for this node instance.

        Logs go to: logs/{NodeName}_{YYYYMMDD_HHMMSS}.log
        """
        node_logger = logging.getLogger(f"4K4D.{self.node_name}")
        node_logger.setLevel(logging.DEBUG)

        # Only add handler if none exist
        if not node_logger.handlers:
            try:
                log_dir = Path(self.env.get_log_dir())
                log_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{self.node_name}_{timestamp}.log"

                handler = logging.FileHandler(str(log_file), mode="w")
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                )
                handler.setFormatter(formatter)
                node_logger.addHandler(handler)
            except Exception:
                # If logging setup fails, don't crash — just use default logger
                pass

        return node_logger

    def _create_runner(self) -> SubprocessRunner:
        """Factory for SubprocessRunner with this node's name and log dir."""
        return SubprocessRunner(self.node_name, self.env.get_log_dir())

    # ── Error Handling ────────────────────────────────────────────────────

    def _safe_execute(self, func, *args, **kwargs) -> Any:
        """
        Wrap node execution in comprehensive error handling.

        NEVER lets exceptions escape to ComfyUI. On any error:
        1. Logs the full traceback to the node's log file
        2. Returns a meaningful error message through the node's outputs
        3. Updates dataset_info with error information if applicable

        This is the ONLY way nodes should execute their logic.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()

            # Log full error
            self._node_logger.error(f"Node {self.node_name} FAILED: {error_msg}")
            self._node_logger.error(f"Traceback:\n{tb}")
            logger.error(f"[{self.node_name}] {error_msg}")

            # Build user-friendly error message
            user_error = self._format_user_error(e)

            # Try to return graceful error through node outputs
            return self._make_error_output(user_error, kwargs)

    def _format_user_error(self, error: Exception) -> str:
        """
        Convert an exception into a user-friendly error message.

        Follows the principle: explain what failed, why, and how to fix it.
        Users have never used 4K4D — no assumed knowledge.
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # Common error patterns with helpful messages
        if isinstance(error, FileNotFoundError):
            return (
                f"FILE NOT FOUND: {error_msg}\n\n"
                "What this means: A required file or directory doesn't exist.\n"
                "How to fix: Check that your input paths are correct and that "
                "previous pipeline steps completed successfully."
            )

        if isinstance(error, PermissionError):
            return (
                f"PERMISSION ERROR: {error_msg}\n\n"
                "What this means: The system doesn't have permission to access a file.\n"
                "How to fix: On RunPod, check that /workspace/ is properly mounted. "
                "Locally, check file permissions."
            )

        if "CUDA" in error_msg or "cuda" in error_msg:
            return (
                f"CUDA ERROR: {error_msg}\n\n"
                "What this means: A GPU/CUDA operation failed.\n"
                "How to fix:\n"
                "1. Check that your GPU has enough free VRAM (4K4D needs ~20GB)\n"
                "2. Try restarting ComfyUI to free GPU memory\n"
                "3. Reduce resolution_scale to 0.5 or lower\n"
                "4. Make sure the DependencyInstall node ran successfully"
            )

        if "out of memory" in error_msg.lower() or "OOM" in error_msg:
            return (
                f"OUT OF MEMORY: {error_msg}\n\n"
                "What this means: The GPU ran out of memory.\n"
                "How to fix:\n"
                "1. Reduce resolution_scale (try 0.25 or 0.5)\n"
                "2. Use fewer training iterations\n"
                "3. Close other GPU-using applications\n"
                "4. Restart ComfyUI to clear VRAM"
            )

        if isinstance(error, ImportError):
            return (
                f"MISSING DEPENDENCY: {error_msg}\n\n"
                "What this means: A required Python package is not installed.\n"
                "How to fix: Run the DependencyInstall node first, or manually "
                "install the missing package with pip."
            )

        # Generic error
        return (
            f"ERROR in {self.node_name}: {error_msg}\n\n"
            f"Error type: {error_type}\n"
            "Check the log file in the logs/ directory for full details.\n"
            "If this keeps happening, please report it as a bug."
        )

    def _make_error_output(self, error_msg: str, kwargs: dict) -> Any:
        """
        Create an error-state output matching the node's RETURN_TYPES.

        Tries to pass error info through dataset_info if available,
        and returns error strings for STRING outputs.
        """
        outputs = []

        for i, return_type in enumerate(self.RETURN_TYPES):
            if return_type == DATASET_INFO_TYPE:
                # Try to get the original dataset_info and add error
                dataset_info = kwargs.get("dataset_info", {})
                if isinstance(dataset_info, dict):
                    updated = dict(dataset_info)
                    updated.setdefault("errors", []).append(error_msg)
                    outputs.append(updated)
                else:
                    outputs.append({"errors": [error_msg]})

            elif return_type == "STRING":
                outputs.append(f"ERROR: {error_msg}")

            elif return_type == "INT":
                outputs.append(0)

            elif return_type == "FLOAT":
                outputs.append(0.0)

            elif return_type == "BOOLEAN":
                outputs.append(False)

            elif return_type == "IMAGE":
                outputs.append(self._create_error_image(error_msg))

            else:
                outputs.append(None)

        return tuple(outputs)

    def _create_error_image(self, error_msg: str):
        """Create a simple red error image placeholder."""
        try:
            import numpy as np
            # Create a 256x512 red-tinted image with error text
            img = np.zeros((256, 512, 3), dtype=np.float32)
            img[:, :, 0] = 0.3  # Red tint
            # Return as [1, H, W, C] tensor (ComfyUI IMAGE format)
            return img[None, :, :, :]
        except Exception:
            return None

    # ── DATASET_INFO Helpers ──────────────────────────────────────────────

    def _validate_dataset_info(self, dataset_info: dict, required_keys: list = None) -> None:
        """
        Validate that dataset_info has required keys.

        Raises:
            ValueError with helpful message if validation fails
        """
        if not isinstance(dataset_info, dict):
            raise ValueError(
                "dataset_info must be a dictionary. Make sure the previous "
                "node in the pipeline is connected correctly."
            )

        if required_keys:
            missing = [k for k in required_keys if k not in dataset_info or dataset_info[k] is None]
            if missing:
                raise ValueError(
                    f"dataset_info is missing required fields: {', '.join(missing)}. "
                    "This usually means a previous pipeline step didn't complete. "
                    "Check that all upstream nodes executed successfully."
                )

    def _update_dataset_info(self, dataset_info: dict, updates: dict) -> dict:
        """
        Return a shallow copy of dataset_info with updates applied.

        NEVER mutates the original dict — this ensures immutability
        through the pipeline.
        """
        new_info = dict(dataset_info)
        new_info.update(updates)
        return new_info

    # ── Caching Control ───────────────────────────────────────────────────

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Default: always re-execute.
        Subclasses can override for caching behavior.
        """
        return float("nan")

    # ── EasyVolcap Command Helpers ────────────────────────────────────────

    def get_easyvolcap_python(self) -> str:
        """Get the Python executable path for EasyVolcap commands."""
        return self.env.python_path

    def build_evc_command(self, command: str, configs: list, extra_args: dict = None) -> list:
        """
        Build an evc-train or evc-test command.

        Args:
            command: "evc-train" or "evc-test"
            configs: List of config file paths to chain with commas
            extra_args: Dict of key=value arguments

        Returns:
            Command list suitable for SubprocessRunner.run()
        """
        cmd = [command, "-c", ",".join(configs)]

        if extra_args:
            for key, value in extra_args.items():
                cmd.append(f"{key}={value}")

        return cmd

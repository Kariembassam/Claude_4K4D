"""
ComfyUI-4K4D Environment Manager
=================================
Singleton that detects the runtime environment (RunPod vs local),
resolves all paths, and provides GPU/CUDA information.

All path resolution goes through this module -- never hardcode /workspace/.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from .constants import (
    DIR_DATA, DIR_DEPS, DIR_LOGS, DIR_EXPORTS, DIR_COMPILED, DIR_SENTINELS,
)

logger = logging.getLogger("4K4D.env_manager")


class EnvManager:
    """
    Singleton environment manager for ComfyUI-4K4D.

    Detects:
    - Whether running on RunPod (via /workspace existence and env vars)
    - GPU capabilities and CUDA version
    - Python path (ComfyUI's Python, not system)
    - All project paths

    Usage:
        env = EnvManager.get_instance()
        data_path = env.paths["data"]
        if env.is_runpod:
            ...
    """

    _instance: Optional["EnvManager"] = None

    @classmethod
    def get_instance(cls) -> "EnvManager":
        """Get or create the singleton EnvManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def __init__(self):
        # ── Runtime Environment Detection ──
        self.is_runpod: bool = self._detect_runpod()
        self.is_serverless: bool = "RUNPOD_SERVERLESS_ID" in os.environ

        # ── Path Resolution ──
        self.node_pack_root: Path = Path(__file__).parent.parent.resolve()
        self.comfyui_root: Path = self._find_comfyui_root()
        self.python_path: str = sys.executable

        # ── Build all paths ──
        self.paths: dict = self._build_paths()

        # ── GPU Info (lazy, filled on first access) ──
        self._gpu_info: Optional[dict] = None
        self._cuda_available: Optional[bool] = None

    def _detect_runpod(self) -> bool:
        """Detect if we're running on RunPod."""
        indicators = [
            Path("/workspace").exists(),
            "RUNPOD_POD_ID" in os.environ,
            "RUNPOD_GPU_COUNT" in os.environ,
        ]
        return any(indicators)

    def _find_comfyui_root(self) -> Path:
        """Find ComfyUI's root directory by walking up from our location."""
        # We're in custom_nodes/ComfyUI-4K4D/core/
        # ComfyUI root is custom_nodes/../../
        candidate = self.node_pack_root.parent.parent
        if (candidate / "main.py").exists() or (candidate / "comfy").is_dir():
            return candidate

        # Fallback: check common locations
        common_paths = [
            Path("/workspace/ComfyUI"),
            Path.home() / "ComfyUI",
            Path("/opt/ComfyUI"),
        ]
        for p in common_paths:
            if p.exists() and (p / "main.py").exists():
                return p

        # Last resort: use parent of custom_nodes
        logger.warning(
            "Could not definitively locate ComfyUI root. "
            f"Using best guess: {candidate}"
        )
        return candidate

    def _build_paths(self) -> dict:
        """Build all project paths based on environment."""
        base = self.node_pack_root

        if self.is_runpod:
            data_root = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-4K4D/data")
            exports_root = Path("/workspace/exports/4k4d")
        else:
            data_root = base / DIR_DATA
            exports_root = base / DIR_EXPORTS

        paths = {
            "node_root": str(base),
            "data": str(data_root),
            "deps": str(base / DIR_DEPS),
            "logs": str(base / DIR_LOGS),
            "exports": str(exports_root),
            "compiled": str(base / DIR_COMPILED),
            "configs": str(base / "configs"),
            "templates": str(base / "configs" / "templates"),
            "workflows": str(base / "workflows"),
            "web": str(base / "web"),
        }

        return paths

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for key in ["data", "deps", "logs", "exports", "compiled"]:
            Path(self.paths[key]).mkdir(parents=True, exist_ok=True)

    def get_dataset_root(self, dataset_name: str) -> str:
        """Get the root directory for a specific dataset."""
        return str(Path(self.paths["data"]) / dataset_name)

    def get_sentinel_dir(self, dataset_name: str) -> str:
        """Get the sentinel directory for a specific dataset."""
        return str(Path(self.paths["data"]) / dataset_name / DIR_SENTINELS)

    def get_log_dir(self) -> str:
        """Get the log directory, creating it if needed."""
        log_dir = Path(self.paths["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

    @property
    def gpu_info(self) -> dict:
        """Get GPU information (lazy loaded)."""
        if self._gpu_info is None:
            self._gpu_info = self._detect_gpu()
        return self._gpu_info

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if self._cuda_available is None:
            try:
                import torch
                self._cuda_available = torch.cuda.is_available()
            except ImportError:
                self._cuda_available = False
        return self._cuda_available

    def _detect_gpu(self) -> dict:
        """Detect GPU capabilities."""
        info = {
            "cuda_available": False,
            "gpu_name": "N/A",
            "gpu_memory_gb": 0.0,
            "cuda_version": "N/A",
            "gpu_arch": "N/A",
            "gpu_count": 0,
        }
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_gb"] = round(props.total_mem / (1024**3), 1)
                info["cuda_version"] = torch.version.cuda or "N/A"
                major, minor = torch.cuda.get_device_capability(0)
                info["gpu_arch"] = f"{major}.{minor}"
                info["gpu_count"] = torch.cuda.device_count()
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

        return info

    def validate_gpu(self, min_vram_gb: float = 20.0) -> tuple:
        """
        Validate that the GPU meets minimum requirements.

        Returns:
            (is_valid, message) tuple
        """
        gpu = self.gpu_info
        if not gpu["cuda_available"]:
            return False, (
                "No CUDA-capable GPU detected. 4K4D requires an NVIDIA GPU "
                "with at least 20GB VRAM. If you're on RunPod, make sure you "
                "selected a GPU-enabled template (RTX 4090 recommended)."
            )

        if gpu["gpu_memory_gb"] < min_vram_gb:
            return False, (
                f"GPU {gpu['gpu_name']} has {gpu['gpu_memory_gb']}GB VRAM, "
                f"but 4K4D requires at least {min_vram_gb}GB. "
                "RTX 4090 (24GB) is the recommended GPU."
            )

        return True, (
            f"GPU OK: {gpu['gpu_name']} with {gpu['gpu_memory_gb']}GB VRAM, "
            f"CUDA {gpu['cuda_version']}, architecture sm_{gpu['gpu_arch'].replace('.', '')}"
        )

    def get_env_info_dict(self) -> dict:
        """Get a summary dict of the environment for DATASET_INFO."""
        return {
            "is_runpod": self.is_runpod,
            "python_path": self.python_path,
            "node_pack_root": str(self.node_pack_root),
            "comfyui_root": str(self.comfyui_root),
            "gpu": self.gpu_info,
        }

    def get_cuda_env_vars(self) -> dict:
        """Get environment variables needed for CUDA compilation."""
        gpu = self.gpu_info
        arch = gpu.get("gpu_arch", "8.6").replace(".", "")
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = f"{gpu.get('gpu_arch', '8.6')}"

        # Find CUDA home
        cuda_home_candidates = [
            os.environ.get("CUDA_HOME", ""),
            "/usr/local/cuda",
            f"/usr/local/cuda-{gpu.get('cuda_version', '11.8')}",
        ]
        for candidate in cuda_home_candidates:
            if candidate and Path(candidate).exists():
                env["CUDA_HOME"] = candidate
                env["PATH"] = f"{candidate}/bin:{env.get('PATH', '')}"
                break

        return env

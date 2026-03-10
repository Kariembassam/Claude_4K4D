"""
ComfyUI-4K4D CUDA Builder
==========================
Lazy CUDA extension compiler with sentinel-file tracking.
Handles compilation of diff-point-rasterization and tinycudann
with automatic retry and fallback behavior.
"""

import logging
from pathlib import Path

from .constants import DIR_COMPILED
from .env_manager import EnvManager
from .subprocess_runner import SubprocessRunner

logger = logging.getLogger("4K4D.cuda_builder")


class CudaBuilder:
    """
    Manages lazy compilation of CUDA extensions.

    Uses sentinel files (.compiled/*.done) to track which extensions
    have been successfully compiled, avoiding redundant compilation.

    Handles:
    - diff-point-rasterization (tile-based CUDA rasterizer)
    - tinycudann (tiny-cuda-nn, hash grid encoding)
    - PyTorch3D (3D operations)
    """

    def __init__(self):
        self.env = EnvManager.get_instance()
        self.compiled_dir = Path(self.env.paths["compiled"])
        self.compiled_dir.mkdir(parents=True, exist_ok=True)
        self.runner = SubprocessRunner("CudaBuilder", self.env.get_log_dir())

    def _sentinel_path(self, extension_name: str) -> Path:
        """Get sentinel file path for an extension."""
        return self.compiled_dir / f"{extension_name}.done"

    def is_compiled(self, extension_name: str) -> bool:
        """Check if an extension has been compiled."""
        return self._sentinel_path(extension_name).exists()

    def mark_compiled(self, extension_name: str) -> None:
        """Mark an extension as compiled."""
        self._sentinel_path(extension_name).write_text("compiled")

    def compile_diff_point_rasterization(self, repo_url: str, commit: str = "main") -> dict:
        """
        Compile diff-point-rasterization CUDA extension.

        Returns:
            dict with keys: success (bool), message (str)
        """
        name = "diff_point_rasterization"
        if self.is_compiled(name):
            return {"success": True, "message": "Already compiled (cached)"}

        logger.info("Compiling diff-point-rasterization...")

        # Try pre-built wheel first
        result = self.runner.run_simple(
            ["pip", "install", f"git+{repo_url}@{commit}"],
            env=self.env.get_cuda_env_vars(),
            timeout=600,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Installed successfully"}

        return {
            "success": False,
            "message": (
                f"Failed to compile diff-point-rasterization. "
                f"This CUDA extension requires nvcc and matching CUDA toolkit. "
                f"Error: {result.error_summary[:500]}"
            ),
        }

    def compile_tinycudann(self) -> dict:
        """
        Install tinycudann (tiny-cuda-nn).
        Tries pre-built wheel first, falls back to source compilation.

        Returns:
            dict with keys: success (bool), message (str)
        """
        name = "tinycudann"
        if self.is_compiled(name):
            return {"success": True, "message": "Already compiled (cached)"}

        logger.info("Installing tinycudann...")

        # Try pre-built wheel first
        result = self.runner.run_simple(
            ["pip", "install", "tinycudann"],
            timeout=120,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Installed from wheel"}

        # Fallback: compile from source
        logger.info("Pre-built wheel failed, compiling from source...")
        result = self.runner.run_simple(
            ["pip", "install", "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"],
            env=self.env.get_cuda_env_vars(),
            timeout=900,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Compiled from source"}

        return {
            "success": False,
            "message": (
                "Failed to install tinycudann. This is a common issue on some systems. "
                "The pipeline will still work but rendering quality may be reduced. "
                f"Error: {result.error_summary[:500]}"
            ),
        }

    def compile_pytorch3d(self) -> dict:
        """
        Install PyTorch3D.
        Tries pre-built wheel, then source compilation.

        Returns:
            dict with keys: success (bool), message (str)
        """
        name = "pytorch3d"
        if self.is_compiled(name):
            return {"success": True, "message": "Already compiled (cached)"}

        logger.info("Installing PyTorch3D...")

        # Try pip install first (may have pre-built wheels)
        result = self.runner.run_simple(
            ["pip", "install", "pytorch3d"],
            timeout=120,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Installed from wheel"}

        # Try with specific index URL for CUDA wheels
        gpu_info = self.env.gpu_info
        cuda_ver = gpu_info.get("cuda_version", "11.8").replace(".", "")

        result = self.runner.run_simple(
            [
                "pip", "install", "pytorch3d",
                "-f", f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu{cuda_ver}_pyt210/download.html",
            ],
            timeout=300,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Installed from CUDA wheel"}

        # Source compilation fallback
        result = self.runner.run_simple(
            ["pip", "install", "git+https://github.com/facebookresearch/pytorch3d.git"],
            env=self.env.get_cuda_env_vars(),
            timeout=1200,
        )

        if result.success:
            self.mark_compiled(name)
            return {"success": True, "message": "Compiled from source"}

        return {
            "success": False,
            "message": (
                "Failed to install PyTorch3D. This is required for 3D operations. "
                f"Error: {result.error_summary[:500]}"
            ),
        }

    def get_compilation_status(self) -> dict:
        """Get compilation status of all tracked extensions."""
        extensions = ["diff_point_rasterization", "tinycudann", "pytorch3d"]
        return {ext: self.is_compiled(ext) for ext in extensions}

    def clear_all(self) -> None:
        """Clear all compilation sentinels (force recompilation)."""
        for path in self.compiled_dir.glob("*.done"):
            path.unlink()
        logger.info("Cleared all compilation sentinels")

"""
ComfyUI-4K4D Node 13: Version Manager
=======================================
Manages version pinning for all dependencies in the 4K4D pipeline.

Provides two modes:
- "stable": Use pinned, tested commit hashes (default, recommended)
- "latest": Use HEAD of main branches (cutting-edge, may break)

This node is typically connected as the version_mode input to the
DependencyInstall node. It can also be used standalone to inspect
the currently installed versions.
"""

import importlib
import logging
import subprocess
from pathlib import Path
from typing import Any

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import (
    CATEGORIES,
    STABLE_PINS,
    LATEST_PINS,
    REPO_URLS,
    __version__,
)

logger = logging.getLogger("4K4D.n13_version_manager")


class FourK4D_VersionManager(BaseEasyVolcapNode):
    """
    Node 13 -- Version Manager

    Controls which versions of EasyVolcap, 4K4D, and CUDA extensions are
    installed by the DependencyInstall node.

    Two modes:
      - stable: Pinned commits that have been tested end-to-end.
                Recommended for production and first-time users.
      - latest: Main branch HEAD. Gets the newest features but may
                introduce breaking changes.

    Outputs:
      - version_mode: "stable" or "latest" (pass to DependencyInstall)
      - version_report: Human-readable report of all versions
      - stable_pins: JSON-like string of pinned commit hashes
      - update_available: True if installed versions differ from pins
    """

    # ── ComfyUI Node Metadata ────────────────────────────────────────────
    CATEGORY = CATEGORIES["setup"]
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("version_mode", "version_report", "stable_pins", "update_available")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version_mode": (["stable", "latest"], {
                    "default": "stable",
                    "tooltip": (
                        "stable: Use tested, pinned versions (recommended).\n"
                        "latest: Use main branch HEAD (cutting-edge, may break)."
                    ),
                }),
            },
            "optional": {
                "show_current_versions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, detect and report currently installed versions "
                        "of all pipeline dependencies."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # ── Execution ────────────────────────────────────────────────────────

    def execute(self, **kwargs) -> Any:
        """Entry point called by ComfyUI. Delegates to _safe_execute."""
        return self._safe_execute(self._run, **kwargs)

    def _run(
        self,
        version_mode: str = "stable",
        show_current_versions: bool = True,
        unique_id: str = None,
    ) -> tuple:
        """
        Main logic for version management.

        Steps:
        1. Select the appropriate pin set (stable or latest)
        2. Optionally detect currently installed versions
        3. Compare installed vs. pinned to determine if updates are available
        4. Generate a human-readable version report
        """
        self._node_logger.info(f"Version manager: mode={version_mode}")

        pins = STABLE_PINS if version_mode == "stable" else LATEST_PINS
        report_lines = []
        update_available = False

        # ── Header ───────────────────────────────────────────────────────
        report_lines.append("=" * 60)
        report_lines.append("4K4D VERSION MANAGER")
        report_lines.append("=" * 60)
        report_lines.append(f"ComfyUI-4K4D version: {__version__}")
        report_lines.append(f"Version mode: {version_mode}")
        report_lines.append("")

        # ── Pinned versions ──────────────────────────────────────────────
        report_lines.append("Pinned versions:")
        for repo_name, pin in pins.items():
            url = REPO_URLS.get(repo_name, "N/A")
            report_lines.append(f"  {repo_name}: {pin}")
            report_lines.append(f"    URL: {url}")
        report_lines.append("")

        # ── Stable pins string ───────────────────────────────────────────
        stable_pins_str = "\n".join(
            f"{name}={commit}" for name, commit in STABLE_PINS.items()
        )

        # ── Current versions (optional) ─────────────────────────────────
        if show_current_versions:
            report_lines.append("Currently installed versions:")
            current_versions = self._detect_installed_versions()

            for pkg_name, version_info in current_versions.items():
                status = version_info.get("status", "unknown")
                version = version_info.get("version", "N/A")
                report_lines.append(f"  {pkg_name}: {version} [{status}]")

                # Check if update available
                if status == "installed" and pkg_name in pins:
                    pinned = pins[pkg_name]
                    if pinned != "main" and version != pinned:
                        update_available = True
                        report_lines.append(f"    -> Update available: {pinned}")

                elif status == "not_installed":
                    update_available = True
                    report_lines.append("    -> Needs installation")

            report_lines.append("")

            # ── Python / PyTorch / CUDA info ─────────────────────────────
            report_lines.append("Environment:")
            report_lines.append(f"  Python: {self._get_python_version()}")
            report_lines.append(f"  PyTorch: {self._get_torch_version()}")
            report_lines.append(f"  CUDA: {self._get_cuda_version()}")
            report_lines.append(f"  GPU: {self.env.gpu_info.get('gpu_name', 'N/A')}")
            report_lines.append(
                f"  VRAM: {self.env.gpu_info.get('gpu_memory_gb', 0)}GB"
            )
            report_lines.append("")

        # ── Summary ──────────────────────────────────────────────────────
        report_lines.append("=" * 60)
        if update_available:
            report_lines.append(
                "STATUS: Updates available. Run DependencyInstall to update."
            )
        else:
            report_lines.append("STATUS: All dependencies are up to date.")
        report_lines.append("=" * 60)

        version_report = "\n".join(report_lines)
        self._node_logger.info(version_report)

        return (version_mode, version_report, stable_pins_str, update_available)

    # ── Version Detection Helpers ────────────────────────────────────────

    def _detect_installed_versions(self) -> dict:
        """
        Detect currently installed versions of all pipeline dependencies.

        Returns a dict mapping package name to version info.
        """
        packages = {
            "easyvolcap": {
                "import_name": "easyvolcap",
                "check_git": True,
            },
            "4k4d": {
                "import_name": None,
                "check_dir": str(Path(self.env.paths["deps"]) / "4K4D"),
            },
            "diff_point_rasterization": {
                "import_name": "diff_point_rasterization",
                "check_git": False,
            },
            "torch": {
                "import_name": "torch",
                "check_git": False,
            },
            "numpy": {
                "import_name": "numpy",
                "check_git": False,
            },
            "robust_video_matting": {
                "import_name": None,
                "check_dir": str(
                    Path(self.env.paths["deps"]) / "RobustVideoMatting"
                ),
            },
        }

        results = {}
        for pkg_name, info in packages.items():
            results[pkg_name] = self._check_package(pkg_name, info)

        return results

    def _check_package(self, name: str, info: dict) -> dict:
        """Check a single package's installation status."""
        result = {"status": "not_installed", "version": "N/A"}

        # Try Python import
        import_name = info.get("import_name")
        if import_name:
            try:
                mod = importlib.import_module(import_name)
                result["status"] = "installed"
                result["version"] = getattr(mod, "__version__", "unknown")
                return result
            except ImportError:
                pass

        # Check if git directory exists
        check_dir = info.get("check_dir")
        if check_dir and Path(check_dir).exists():
            result["status"] = "installed"
            result["version"] = self._get_git_commit(check_dir)
            return result

        return result

    def _get_git_commit(self, repo_dir: str) -> str:
        """Get the current git commit hash of a repo directory."""
        try:
            git_dir = Path(repo_dir) / ".git"
            if not git_dir.exists():
                return "unknown (not a git repo)"

            proc = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=5,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_torch_version(self) -> str:
        """Get PyTorch version string."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "not installed"

    def _get_cuda_version(self) -> str:
        """Get CUDA version from PyTorch."""
        try:
            import torch
            return torch.version.cuda or "N/A"
        except ImportError:
            return "N/A"

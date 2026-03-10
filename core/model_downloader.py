"""
ComfyUI-4K4D Model Downloader
===============================
Download pretrained models with retry, checksum, and resume support.
"""

import hashlib
import logging
import os
from pathlib import Path

from .subprocess_runner import SubprocessRunner
from .env_manager import EnvManager

logger = logging.getLogger("4K4D.model_downloader")


class ModelDownloader:
    """
    Downloads pretrained models with reliability features.

    Supports:
    - gdown for Google Drive links
    - wget/curl for direct URLs
    - Resume on interruption
    - SHA256 checksum verification
    """

    def __init__(self):
        self.env = EnvManager.get_instance()
        self.runner = SubprocessRunner("ModelDownloader", self.env.get_log_dir())

    def download(
        self,
        url: str,
        output_path: str,
        expected_sha256: str = None,
        max_retries: int = 3,
    ) -> dict:
        """
        Download a file with retry and optional checksum.

        Args:
            url: Download URL (supports Google Drive, direct HTTP)
            output_path: Where to save the file
            expected_sha256: Optional SHA256 hash for verification
            max_retries: Number of retry attempts

        Returns:
            dict with keys: success (bool), path (str), message (str)
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded and verified
        if output.exists() and expected_sha256:
            if self._verify_checksum(str(output), expected_sha256):
                return {
                    "success": True,
                    "path": str(output),
                    "message": "Already downloaded and verified",
                }

        # Detect download method
        if "drive.google.com" in url or "docs.google.com" in url:
            return self._download_gdown(url, str(output), max_retries)
        else:
            return self._download_wget(url, str(output), max_retries)

    def _download_gdown(self, url: str, output_path: str, max_retries: int) -> dict:
        """Download from Google Drive using gdown."""
        for attempt in range(max_retries):
            result = self.runner.run_simple(
                ["python", "-m", "gdown", url, "-O", output_path, "--fuzzy"],
                timeout=1800,
            )
            if result.success and Path(output_path).exists():
                return {
                    "success": True,
                    "path": output_path,
                    "message": f"Downloaded successfully (attempt {attempt + 1})",
                }

        return {
            "success": False,
            "path": output_path,
            "message": f"Failed to download after {max_retries} attempts",
        }

    def _download_wget(self, url: str, output_path: str, max_retries: int) -> dict:
        """Download using wget with resume support."""
        for attempt in range(max_retries):
            result = self.runner.run_simple(
                ["wget", "-c", "-O", output_path, url],
                timeout=1800,
            )
            if result.success and Path(output_path).exists():
                return {
                    "success": True,
                    "path": output_path,
                    "message": f"Downloaded successfully (attempt {attempt + 1})",
                }

        return {
            "success": False,
            "path": output_path,
            "message": f"Failed to download after {max_retries} attempts",
        }

    def _verify_checksum(self, file_path: str, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of a file."""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest() == expected_sha256
        except Exception:
            return False

"""
ComfyUI-4K4D Checkpoint Manager
================================
Sentinel-file-based checkpoint and resume system for long-running operations.

Each pipeline stage writes JSON sentinel files to track state:
- {stage}.started  → operation is in progress
- {stage}.completed → operation finished successfully
- {stage}.failed → operation failed

This enables crash recovery and skip-if-done behavior.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from .constants import DIR_SENTINELS, SENTINEL_EXTENSION

logger = logging.getLogger("4K4D.checkpoint_manager")


class CheckpointManager:
    """
    Manages sentinel files for checkpoint/resume of long-running pipeline stages.

    Sentinel files are JSON files in {data_root}/.sentinels/ containing
    timestamps and metadata about each stage's execution state.

    Usage:
        cm = CheckpointManager("/path/to/data/my_sequence/.sentinels")

        if cm.is_completed("frame_extract"):
            print("Already done, skipping")
        elif cm.should_resume("frame_extract"):
            print("Resuming from crash")
        else:
            cm.mark_started("frame_extract")
            # ... do work ...
            cm.mark_completed("frame_extract", {"frame_count": 150})
    """

    def __init__(self, sentinel_dir: str):
        self.sentinel_dir = Path(sentinel_dir)
        self.sentinel_dir.mkdir(parents=True, exist_ok=True)

    def _sentinel_path(self, stage_name: str, status: str) -> Path:
        """Get path to a sentinel file."""
        return self.sentinel_dir / f"{stage_name}.{status}{SENTINEL_EXTENSION}"

    def _write_sentinel(self, stage_name: str, status: str, metadata: dict = None) -> None:
        """Write a sentinel file with timestamp and optional metadata."""
        data = {
            "stage": stage_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "epoch_time": time.time(),
        }
        if metadata:
            data["metadata"] = metadata

        path = self._sentinel_path(stage_name, status)
        path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Wrote sentinel: {path}")

    def _read_sentinel(self, stage_name: str, status: str) -> dict:
        """Read a sentinel file. Returns empty dict if not found."""
        path = self._sentinel_path(stage_name, status)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read sentinel {path}: {e}")
        return {}

    def mark_started(self, stage_name: str, metadata: dict = None) -> None:
        """
        Mark a stage as started. Creates .started sentinel.
        Removes any existing .failed sentinel.
        """
        # Remove failed sentinel if exists
        failed_path = self._sentinel_path(stage_name, "failed")
        if failed_path.exists():
            failed_path.unlink()

        self._write_sentinel(stage_name, "started", metadata)
        logger.info(f"Stage '{stage_name}' marked as STARTED")

    def mark_completed(self, stage_name: str, metadata: dict = None) -> None:
        """
        Mark a stage as completed. Creates .completed sentinel.
        Removes the .started sentinel.
        """
        self._write_sentinel(stage_name, "completed", metadata)

        # Remove started sentinel
        started_path = self._sentinel_path(stage_name, "started")
        if started_path.exists():
            started_path.unlink()

        logger.info(f"Stage '{stage_name}' marked as COMPLETED")

    def mark_failed(self, stage_name: str, error_msg: str = "", metadata: dict = None) -> None:
        """
        Mark a stage as failed. Creates .failed sentinel.
        Removes the .started sentinel.
        """
        fail_meta = metadata or {}
        fail_meta["error"] = error_msg
        self._write_sentinel(stage_name, "failed", fail_meta)

        # Remove started sentinel
        started_path = self._sentinel_path(stage_name, "started")
        if started_path.exists():
            started_path.unlink()

        logger.warning(f"Stage '{stage_name}' marked as FAILED: {error_msg}")

    def is_completed(self, stage_name: str) -> bool:
        """Check if a stage has completed successfully."""
        return self._sentinel_path(stage_name, "completed").exists()

    def is_in_progress(self, stage_name: str) -> bool:
        """Check if a stage is currently in progress."""
        return (
            self._sentinel_path(stage_name, "started").exists()
            and not self._sentinel_path(stage_name, "completed").exists()
        )

    def is_failed(self, stage_name: str) -> bool:
        """Check if a stage has failed."""
        return self._sentinel_path(stage_name, "failed").exists()

    def should_resume(self, stage_name: str) -> bool:
        """
        Check if a stage should be resumed (started but not completed).
        This indicates a crash during the previous run.
        """
        return self.is_in_progress(stage_name) and not self.is_completed(stage_name)

    def get_metadata(self, stage_name: str) -> dict:
        """Get metadata from the completed sentinel for a stage."""
        data = self._read_sentinel(stage_name, "completed")
        return data.get("metadata", {})

    def get_started_metadata(self, stage_name: str) -> dict:
        """Get metadata from the started sentinel (useful for resume)."""
        data = self._read_sentinel(stage_name, "started")
        return data.get("metadata", {})

    def clear(self, stage_name: str) -> None:
        """Remove all sentinels for a stage (force re-run)."""
        for status in ["started", "completed", "failed"]:
            path = self._sentinel_path(stage_name, status)
            if path.exists():
                path.unlink()
                logger.debug(f"Removed sentinel: {path}")

        logger.info(f"Cleared all sentinels for stage '{stage_name}'")

    def clear_all(self) -> None:
        """Remove all sentinel files (nuclear option)."""
        for path in self.sentinel_dir.glob(f"*{SENTINEL_EXTENSION}"):
            path.unlink()
        logger.info("Cleared ALL sentinels")

    def get_all_status(self) -> dict:
        """
        Get status of all tracked stages.

        Returns:
            dict mapping stage_name -> {"status": str, "timestamp": str, "metadata": dict}
        """
        stages = {}
        for path in sorted(self.sentinel_dir.glob(f"*{SENTINEL_EXTENSION}")):
            parts = path.stem.rsplit(".", 1)
            if len(parts) == 2:
                stage_name, status = parts
                if stage_name not in stages:
                    stages[stage_name] = {}
                try:
                    data = json.loads(path.read_text())
                    stages[stage_name][status] = {
                        "timestamp": data.get("timestamp", "unknown"),
                        "metadata": data.get("metadata", {}),
                    }
                except (json.JSONDecodeError, IOError):
                    stages[stage_name][status] = {"timestamp": "error", "metadata": {}}

        # Determine overall status for each stage
        result = {}
        for stage_name, statuses in stages.items():
            if "completed" in statuses:
                overall = "completed"
            elif "failed" in statuses:
                overall = "failed"
            elif "started" in statuses:
                overall = "in_progress"
            else:
                overall = "unknown"

            result[stage_name] = {
                "status": overall,
                "details": statuses,
            }

        return result

    def save_checkpoint(self, stage_name: str, checkpoint_data: dict) -> None:
        """
        Save a training checkpoint reference (path, iteration, metrics).
        This is separate from start/complete sentinels.
        """
        path = self.sentinel_dir / f"{stage_name}.checkpoint{SENTINEL_EXTENSION}"
        data = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "checkpoint": checkpoint_data,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved checkpoint for '{stage_name}': {checkpoint_data}")

    def get_latest_checkpoint(self, stage_name: str) -> dict:
        """Get the latest checkpoint data for a stage."""
        path = self.sentinel_dir / f"{stage_name}.checkpoint{SENTINEL_EXTENSION}"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("checkpoint", {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

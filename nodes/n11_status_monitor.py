"""
ComfyUI-4K4D Node 11: Status Monitor
======================================
Live training progress viewer with log tail, metrics, and disk usage.
"""

import logging
import os
import time
from pathlib import Path

from core.base_node import BaseEasyVolcapNode
from core.constants import CATEGORIES, DATASET_INFO_TYPE
from core.checkpoint_manager import CheckpointManager

logger = logging.getLogger("4K4D.n11_status_monitor")


class FourK4D_StatusMonitor(BaseEasyVolcapNode):
    """
    Live monitoring dashboard for the 4K4D pipeline.

    Shows:
    - Last N lines of training logs
    - Current PSNR and loss values
    - Estimated time remaining
    - Training phase status
    - Disk usage
    """

    CATEGORY = CATEGORIES["utilities"]
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "IMAGE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = (
        "status_text", "current_iteration", "current_psnr",
        "loss_curve_image", "estimated_time_remaining",
        "training_phase", "disk_usage_gb"
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "log_tail_lines": ("INT", {"default": 50, "min": 10, "max": 500}),
                "refresh_on_execute": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute to get fresh status."""
        return float("nan")

    def execute(self, dataset_info, log_tail_lines=50, refresh_on_execute=True,
                unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, log_tail_lines, refresh_on_execute, unique_id
        )

    def _run(self, dataset_info, log_tail_lines, refresh_on_execute, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root"])

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info.get("dataset_name", "unknown")
        status_lines = []
        current_iter = 0
        current_psnr = 0.0
        phase = "unknown"
        eta = "N/A"

        # Check sentinel status
        sentinel_dir = os.path.join(dataset_root, ".sentinels")
        if os.path.exists(sentinel_dir):
            cm = CheckpointManager(sentinel_dir)
            all_status = cm.get_all_status()

            status_lines.append("=" * 50)
            status_lines.append(f"PIPELINE STATUS: {name}")
            status_lines.append("=" * 50)

            stages = [
                "frame_extract", "camera_calibration", "mask_generation",
                "visual_hull", "training", "supercharge", "render"
            ]
            for stage in stages:
                if stage in all_status:
                    info = all_status[stage]
                    icon = {"completed": "[OK]", "in_progress": "[>>]", "failed": "[X]", "unknown": "[?]"}
                    status_lines.append(f"  {icon.get(info['status'], '[?]')} {stage}: {info['status']}")
                    if info["status"] == "in_progress":
                        phase = stage
                else:
                    status_lines.append(f"  [ ] {stage}: not started")

        # Read training logs
        log_dir = self.env.get_log_dir()
        training_logs = sorted(
            Path(log_dir).glob("*Train*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )

        if training_logs:
            latest_log = training_logs[0]
            status_lines.append(f"\n--- Latest training log: {latest_log.name} ---")

            try:
                with open(latest_log) as f:
                    lines = f.readlines()
                    tail = lines[-log_tail_lines:]
                    for line in tail:
                        status_lines.append(line.rstrip())

                        # Parse iteration and PSNR
                        import re
                        iter_match = re.search(r'[Ii]ter[:\s]+(\d+)', line)
                        if iter_match:
                            current_iter = int(iter_match.group(1))

                        psnr_match = re.search(r'[Pp][Ss][Nn][Rr][:\s]+([\d.]+)', line)
                        if psnr_match:
                            current_psnr = float(psnr_match.group(1))
            except Exception as e:
                status_lines.append(f"Failed to read log: {e}")

        # Estimate time remaining
        if current_iter > 0:
            max_iter = dataset_info.get("max_iterations", 1600)
            if current_iter < max_iter:
                # Simple linear estimate
                elapsed = time.time()  # This would need a start time reference
                progress = current_iter / max_iter
                if progress > 0.01:
                    eta = f"~{int((1.0 - progress) / progress * 100)}% remaining"
            else:
                eta = "Complete"
                phase = "complete"

        # Calculate disk usage
        disk_usage_gb = self._get_dir_size_gb(dataset_root)
        status_lines.append(f"\nDisk usage: {disk_usage_gb:.2f} GB")

        # Create a simple loss curve image
        loss_curve = self._create_error_image(f"PSNR: {current_psnr:.2f} dB | Iter: {current_iter}")

        return (
            "\n".join(status_lines),
            current_iter,
            current_psnr,
            loss_curve,
            eta,
            phase,
            disk_usage_gb,
        )

    def _get_dir_size_gb(self, path):
        """Calculate total directory size in GB."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        except Exception:
            pass
        return total / (1024 ** 3)

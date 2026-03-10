"""
ComfyUI-4K4D Node 14: Cleanup
================================
Manual cleanup of intermediate files. User runs when satisfied with results.
SAFETY: dry_run=True by default. Never deletes trained model files.
"""

import logging
import os
import shutil
import tarfile
from pathlib import Path

from core.base_node import BaseEasyVolcapNode
from core.constants import CATEGORIES, DATASET_INFO_TYPE

logger = logging.getLogger("4K4D.n14_cleanup")


class FourK4D_Cleanup(BaseEasyVolcapNode):
    """
    Selective cleanup of intermediate pipeline files.

    Safety features:
    - dry_run=True by default (shows what WOULD be deleted)
    - Never deletes trained model or supercharged model files
    - Optional archive-before-delete
    - Shows disk usage before/after preview
    """

    CATEGORY = CATEGORIES["utilities"]
    FUNCTION = "execute"
    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("space_freed_gb", "archive_path", "deletion_report", "remaining_size_gb")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "delete_raw_frames": ("BOOLEAN", {"default": False}),
                "delete_masks": ("BOOLEAN", {"default": False}),
                "delete_vhulls": ("BOOLEAN", {"default": False}),
                "delete_training_logs": ("BOOLEAN", {"default": False}),
                "archive_before_delete": ("BOOLEAN", {"default": True}),
                "dry_run": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, delete_raw_frames=False, delete_masks=False,
                delete_vhulls=False, delete_training_logs=False,
                archive_before_delete=True, dry_run=True, unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, delete_raw_frames, delete_masks,
            delete_vhulls, delete_training_logs, archive_before_delete,
            dry_run, unique_id
        )

    def _run(self, dataset_info, delete_raw_frames, delete_masks,
             delete_vhulls, delete_training_logs, archive_before_delete,
             dry_run, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root"])

        dataset_root = dataset_info["dataset_root"]
        report_lines = []
        total_freed = 0.0
        archive_path = ""

        # Calculate sizes of directories to delete
        targets = []
        if delete_raw_frames:
            targets.append(("images", os.path.join(dataset_root, "images")))
        if delete_masks:
            targets.append(("masks", os.path.join(dataset_root, "masks")))
        if delete_vhulls:
            targets.append(("vhulls", os.path.join(dataset_root, "vhulls")))
            targets.append(("surfs", os.path.join(dataset_root, "surfs")))
        if delete_training_logs:
            targets.append(("logs", self.env.get_log_dir()))

        report_lines.append("=" * 50)
        report_lines.append(f"CLEANUP {'PREVIEW (DRY RUN)' if dry_run else 'EXECUTION'}")
        report_lines.append("=" * 50)

        if not targets:
            report_lines.append("No directories selected for cleanup.")
            remaining = self._dir_size_gb(dataset_root)
            return (0.0, "", "\n".join(report_lines), remaining)

        for name, path in targets:
            if os.path.exists(path):
                size_gb = self._dir_size_gb(path)
                total_freed += size_gb
                report_lines.append(f"  [{name}] {path}")
                report_lines.append(f"    Size: {size_gb:.3f} GB")
                if dry_run:
                    report_lines.append(f"    Action: WOULD DELETE")
                else:
                    report_lines.append(f"    Action: DELETING")
            else:
                report_lines.append(f"  [{name}] Not found: {path}")

        report_lines.append(f"\nTotal space to free: {total_freed:.3f} GB")

        if dry_run:
            report_lines.append("\nDRY RUN — Nothing was deleted.")
            report_lines.append("Set dry_run=False to actually delete files.")
        else:
            # Archive first if requested
            if archive_before_delete:
                archive_name = f"{dataset_info.get('dataset_name', 'data')}_cleanup_archive.tar.gz"
                archive_path = os.path.join(dataset_root, archive_name)
                report_lines.append(f"\nArchiving to: {archive_path}")

                try:
                    with tarfile.open(archive_path, "w:gz") as tar:
                        for name, path in targets:
                            if os.path.exists(path):
                                tar.add(path, arcname=name)
                    report_lines.append("Archive created successfully.")
                except Exception as e:
                    report_lines.append(f"Archive FAILED: {e}")
                    report_lines.append("Aborting deletion for safety.")
                    remaining = self._dir_size_gb(dataset_root)
                    return (0.0, "", "\n".join(report_lines), remaining)

            # Delete directories
            for name, path in targets:
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        report_lines.append(f"  DELETED: {path}")
                    except Exception as e:
                        report_lines.append(f"  FAILED to delete {path}: {e}")

            report_lines.append(f"\nSpace freed: {total_freed:.3f} GB")

        remaining = self._dir_size_gb(dataset_root)
        report_lines.append(f"Remaining dataset size: {remaining:.3f} GB")

        # Safety notice
        report_lines.append("\nSAFETY: Trained model and supercharged model files are NEVER deleted.")

        return (
            total_freed if not dry_run else 0.0,
            archive_path,
            "\n".join(report_lines),
            remaining,
        )

    def _dir_size_gb(self, path):
        """Calculate directory size in GB."""
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

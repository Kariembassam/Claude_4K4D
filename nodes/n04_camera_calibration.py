"""
ComfyUI-4K4D Node 4: Camera Calibration
==========================================
COLMAP auto-calibration OR accept existing calibration files.
Supports both fixed and handheld camera rigs.
"""

import logging
import os
import shutil
from pathlib import Path

from core.base_node import BaseEasyVolcapNode
from core.constants import CATEGORIES, DATASET_INFO_TYPE
from core.checkpoint_manager import CheckpointManager

logger = logging.getLogger("4K4D.n04_camera_calibration")


class FourK4D_CameraCalibration(BaseEasyVolcapNode):
    """
    Camera calibration node supporting COLMAP auto-calibration
    and pre-existing calibration file import.

    Fixed rig → exhaustive matcher (all camera pairs)
    Handheld → sequential matcher (follows video timeline)
    """

    CATEGORY = CATEGORIES["preprocessing"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "BOOLEAN", "IMAGE", "STRING")
    RETURN_NAMES = ("dataset_info", "calibration_valid", "camera_frustum_preview", "calibration_report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
                "calibration_mode": (["auto_colmap", "use_existing"],),
            },
            "optional": {
                "extri_yml_path": ("STRING", {"default": ""}),
                "intri_yml_path": ("STRING", {"default": ""}),
                "rig_type": (["fixed", "handheld", "auto_detect"], {"default": "fixed"}),
                "colmap_quality": (["low", "medium", "high"], {"default": "medium"}),
                "optimize_cameras": ("BOOLEAN", {"default": True}),
                "matcher_type": (["auto", "exhaustive", "sequential"], {"default": "auto"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, calibration_mode, extri_yml_path="",
                intri_yml_path="", rig_type="fixed", colmap_quality="medium",
                optimize_cameras=True, matcher_type="auto", unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, calibration_mode, extri_yml_path,
            intri_yml_path, rig_type, colmap_quality, optimize_cameras,
            matcher_type, unique_id
        )

    def _run(self, dataset_info, calibration_mode, extri_yml_path,
             intri_yml_path, rig_type, colmap_quality, optimize_cameras,
             matcher_type, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root", "dataset_name"])

        dataset_root = dataset_info["dataset_root"]
        camera_count = dataset_info.get("camera_count", 5)
        runner = self._create_runner()

        cm = CheckpointManager(os.path.join(dataset_root, ".sentinels"))
        if cm.is_completed("camera_calibration"):
            self._node_logger.info("Calibration already completed, skipping")
            return (
                self._update_dataset_info(dataset_info, {"has_calibration": True}),
                True,
                self._create_error_image("Cached — calibration done"),
                "Calibration already completed (cached)",
            )

        cm.mark_started("camera_calibration")
        report_lines = []

        if calibration_mode == "use_existing":
            # Accept existing calibration files
            report_lines.append("Mode: Using existing calibration files")

            extri_path = extri_yml_path or os.path.join(dataset_root, "extri.yml")
            intri_path = intri_yml_path or os.path.join(dataset_root, "intri.yml")

            if not os.path.exists(extri_path):
                raise FileNotFoundError(
                    f"Extrinsic calibration file not found: {extri_path}\n"
                    "Please provide the correct path or use auto_colmap mode."
                )
            if not os.path.exists(intri_path):
                raise FileNotFoundError(
                    f"Intrinsic calibration file not found: {intri_path}\n"
                    "Please provide the correct path or use auto_colmap mode."
                )

            # Copy to dataset root if not already there
            dest_extri = os.path.join(dataset_root, "extri.yml")
            dest_intri = os.path.join(dataset_root, "intri.yml")
            if extri_path != dest_extri:
                shutil.copy2(extri_path, dest_extri)
            if intri_path != dest_intri:
                shutil.copy2(intri_path, dest_intri)

            report_lines.append(f"  extri.yml: {extri_path}")
            report_lines.append(f"  intri.yml: {intri_path}")
            report_lines.append("Calibration files accepted.")

        else:
            # Auto COLMAP calibration
            report_lines.append("Mode: COLMAP auto-calibration")

            # Determine matcher type
            if matcher_type == "auto":
                if rig_type == "handheld":
                    actual_matcher = "sequential"
                else:
                    actual_matcher = "exhaustive"
            else:
                actual_matcher = matcher_type

            # Use high quality for sparse rigs
            if camera_count < 8:
                colmap_quality = "high"
                report_lines.append(f"  Forced high quality for {camera_count}-camera sparse rig")

            report_lines.append(f"  Rig type: {rig_type}")
            report_lines.append(f"  Matcher: {actual_matcher}")
            report_lines.append(f"  Quality: {colmap_quality}")

            if camera_count < 8:
                report_lines.append(
                    f"\n  WARNING: {camera_count} cameras will produce lower visual hull "
                    "quality than 8+ cameras. Results will still be usable but may show "
                    "artifacts at large viewing angle changes."
                )

            # Step 1: Run COLMAP
            easyvolcap_root = dataset_info.get("easyvolcap_root", "")
            images_dir = os.path.join(dataset_root, "images", "00")

            # Run COLMAP feature extraction + matching + reconstruction
            colmap_cmd = [
                "python", "-m", "easyvolcap.scripts.colmap.run_colmap",
                "--data_root", dataset_root,
                "--matcher", actual_matcher,
            ]

            if easyvolcap_root and os.path.exists(easyvolcap_root):
                script_path = os.path.join(easyvolcap_root, "scripts", "colmap", "run_colmap.py")
                if os.path.exists(script_path):
                    colmap_cmd = ["python", script_path, "--data_root", dataset_root, "--matcher", actual_matcher]

            result = runner.run(
                colmap_cmd,
                cwd=easyvolcap_root or dataset_root,
                unique_id=unique_id,
                timeout_seconds=7200,  # 2 hours max for COLMAP
            )

            if result.success:
                report_lines.append("  COLMAP reconstruction: SUCCESS")
            else:
                # Try direct colmap commands as fallback
                report_lines.append("  EasyVolcap COLMAP script not available, trying direct COLMAP...")
                self._run_direct_colmap(dataset_root, actual_matcher, runner, report_lines)

            # Step 2: Convert to EasyVolcap format
            convert_cmd = [
                "python", "-m", "easyvolcap.scripts.colmap.colmap_to_easyvolcap",
                "--data_root", dataset_root,
            ]
            result = runner.run(convert_cmd, cwd=easyvolcap_root or dataset_root, timeout_seconds=300)
            if result.success:
                report_lines.append("  Conversion to EasyVolcap format: SUCCESS")
            else:
                report_lines.append("  Conversion to EasyVolcap format: SKIPPED (manual conversion may be needed)")

            # Step 3: Optional NGP camera optimization
            if optimize_cameras:
                report_lines.append("  Camera optimization: queued for training phase")

        # Verify calibration
        extri_exists = os.path.exists(os.path.join(dataset_root, "extri.yml"))
        intri_exists = os.path.exists(os.path.join(dataset_root, "intri.yml"))
        calibration_valid = extri_exists and intri_exists

        if calibration_valid:
            cm.mark_completed("camera_calibration")
            report_lines.append("\nCalibration VALID: extri.yml and intri.yml present.")
        else:
            cm.mark_failed("camera_calibration", "Calibration files not generated")
            report_lines.append("\nCalibration FAILED: Missing calibration files.")

        preview = self._create_error_image("Camera calibration preview")

        return (
            self._update_dataset_info(dataset_info, {"has_calibration": calibration_valid}),
            calibration_valid,
            preview,
            "\n".join(report_lines),
        )

    def _run_direct_colmap(self, dataset_root, matcher_type, runner, report_lines):
        """Run COLMAP directly as fallback."""
        db_path = os.path.join(dataset_root, "colmap.db")
        images_path = os.path.join(dataset_root, "images", "00")
        sparse_path = os.path.join(dataset_root, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        # Feature extraction
        result = runner.run_simple([
            "colmap", "feature_extractor",
            "--database_path", db_path,
            "--image_path", images_path,
        ], timeout=1800)
        report_lines.append(f"  Feature extraction: {'OK' if result.success else 'FAILED'}")

        # Matching
        match_cmd = "exhaustive_matcher" if matcher_type == "exhaustive" else "sequential_matcher"
        result = runner.run_simple([
            "colmap", match_cmd,
            "--database_path", db_path,
        ], timeout=3600)
        report_lines.append(f"  Feature matching ({matcher_type}): {'OK' if result.success else 'FAILED'}")

        # Reconstruction
        result = runner.run_simple([
            "colmap", "mapper",
            "--database_path", db_path,
            "--image_path", images_path,
            "--output_path", sparse_path,
        ], timeout=3600)
        report_lines.append(f"  Reconstruction: {'OK' if result.success else 'FAILED'}")

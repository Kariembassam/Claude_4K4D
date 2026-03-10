"""
ComfyUI-4K4D Node 01: Folder Ingest
=====================================
Entry point for the 4K4D pipeline. Validates a user-provided folder
containing multi-camera video/image data, detects its structure,
and builds the initial DATASET_INFO dict that flows through every
subsequent node.

Supports three input layouts:
1. Raw videos  -- folder with N video files (one per camera)
2. EasyVolcap  -- folder with images/{00,01,...}/000000.jpg structure
3. Flat images -- folder with image files (limited support)

Also detects the presence of:
- Calibration files (extri.yml + intri.yml)
- Pre-existing masks (masks/{00,01,...}/)
- Pre-existing visual hulls

The node creates the full EasyVolcap directory structure under
data/{dataset_name}/ so that downstream nodes have a consistent
workspace.
"""

import logging
import os
from pathlib import Path
from typing import Any

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import (
    CATEGORIES,
    DATASET_INFO_TYPE,
    DEFAULTS,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    create_empty_dataset_info,
)
from ..core.format_detector import FormatDetector
from ..core.dataset_structure import create_dataset_dirs, validate_dataset_structure

logger = logging.getLogger("4K4D.n01_folder_ingest")


class FourK4D_FolderIngest(BaseEasyVolcapNode):
    """
    Node 01 -- Folder Ingest

    Accepts a path to a folder of multi-camera footage and validates it.
    Produces the DATASET_INFO dict consumed by every downstream node.

    This is the FIRST node in the pipeline and does not require
    DATASET_INFO as input (it creates it from scratch).

    Typical usage:
        1. Point video_folder at a directory containing N video files
           (cam00.mp4, cam01.mp4, ...) or an EasyVolcap-format directory
        2. Provide a dataset_name (used for all downstream output paths)
        3. Optionally set camera_count if auto-detection is wrong
        4. Connect outputs to DependencyInstall and FrameExtract nodes
    """

    # ── ComfyUI Node Metadata ────────────────────────────────────────────
    CATEGORY = CATEGORIES["input"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("dataset_info", "has_calibration", "has_masks", "validation_report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Absolute path to the folder containing your multi-camera "
                        "footage. Accepts: a folder of video files (one per camera), "
                        "an EasyVolcap-format directory with images/ subdirectories, "
                        "or a folder of images."
                    ),
                }),
                "dataset_name": ("STRING", {
                    "default": DEFAULTS["dataset_name"],
                    "tooltip": (
                        "A short name for this capture sequence. Used as the output "
                        "directory name under data/. Example: 'dancer_01'"
                    ),
                }),
            },
            "optional": {
                "camera_count": ("INT", {
                    "default": DEFAULTS["camera_count"],
                    "min": 2,
                    "max": 60,
                    "tooltip": (
                        "Number of cameras in your rig. Auto-detected from the "
                        "number of video files or image subdirectories, but you "
                        "can override it here."
                    ),
                }),
                "expected_fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 240.0,
                    "step": 0.001,
                    "tooltip": (
                        "Expected frame rate. Set to 0.0 for auto-detection. "
                        "If set, the node will warn if detected FPS differs."
                    ),
                }),
                "strict_validation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, the node will fail on any inconsistency "
                        "(resolution mismatch, FPS mismatch, missing files). "
                        "If False, it will warn but continue."
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
        video_folder: str,
        dataset_name: str,
        camera_count: int = DEFAULTS["camera_count"],
        expected_fps: float = 0.0,
        strict_validation: bool = True,
        unique_id: str = None,
    ) -> tuple:
        """
        Main logic for folder ingest.

        Steps:
        1. Validate the input folder exists and is not empty
        2. Use FormatDetector to analyze folder contents
        3. If videos are found, validate consistency (resolution, FPS, codec)
        4. Build the DATASET_INFO dict
        5. Create the EasyVolcap directory structure
        6. Return all outputs
        """
        self._node_logger.info(f"Starting folder ingest: {video_folder}")
        report_lines = []

        # ── Step 1: Basic folder validation ──────────────────────────────
        folder = Path(video_folder)
        if not video_folder or not video_folder.strip():
            raise ValueError(
                "video_folder is empty. Please provide the path to your "
                "multi-camera footage folder."
            )

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder not found: {video_folder}\n\n"
                "Make sure the path is correct. On RunPod, data is usually "
                "at /workspace/data/. Locally, use an absolute path like "
                "/home/user/captures/my_sequence/"
            )

        if not folder.is_dir():
            raise ValueError(
                f"Path is not a directory: {video_folder}\n"
                "Please provide a folder path, not a file path."
            )

        report_lines.append(f"Input folder: {video_folder}")
        report_lines.append(f"Dataset name: {dataset_name}")

        # ── Step 2: Detect folder contents ───────────────────────────────
        detector = FormatDetector()
        contents = detector.detect_folder_contents(video_folder)

        structure_type = contents["structure_type"]
        detected_cameras = contents["camera_count"]
        has_calibration = contents["has_calibration"]
        has_masks = contents["has_masks"]

        report_lines.append(f"Structure type: {structure_type}")
        report_lines.append(f"Detected cameras: {detected_cameras}")
        report_lines.append(f"Has calibration: {has_calibration}")
        report_lines.append(f"Has masks: {has_masks}")
        report_lines.append(f"Has vhulls: {contents['has_vhulls']}")

        if structure_type == "unknown":
            raise ValueError(
                f"Could not detect any video or image files in {video_folder}.\n\n"
                "Expected one of:\n"
                "  1. A folder with video files (cam00.mp4, cam01.mp4, ...)\n"
                "  2. An EasyVolcap folder with images/00/, images/01/, ...\n"
                "  3. A folder with image files (.jpg, .png)\n\n"
                f"Supported video formats: {', '.join(sorted(SUPPORTED_VIDEO_FORMATS))}\n"
                f"Supported image formats: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
            )

        # Use detected camera count if it makes sense, otherwise use user input
        if detected_cameras > 0:
            if detected_cameras != camera_count:
                msg = (
                    f"Auto-detected {detected_cameras} cameras, but camera_count "
                    f"is set to {camera_count}. Using detected count ({detected_cameras})."
                )
                report_lines.append(f"WARNING: {msg}")
                self._node_logger.warning(msg)
            actual_camera_count = detected_cameras
        else:
            actual_camera_count = camera_count
            report_lines.append(
                f"Could not auto-detect camera count. Using provided value: {camera_count}"
            )

        # ── Step 3: Validate video consistency (if video-based) ──────────
        detected_fps = 0.0
        detected_resolution = ""
        detected_format = ""
        video_paths = contents["video_files"]

        if structure_type == "videos" and video_paths:
            report_lines.append(f"\nValidating {len(video_paths)} video files...")

            consistency = detector.validate_videos_consistency(video_paths)

            detected_fps = consistency.get("fps", 0.0)
            detected_resolution = consistency.get("resolution", "")
            detected_format = consistency.get("codec", "")

            report_lines.append(f"  FPS: {detected_fps}")
            report_lines.append(f"  Resolution: {detected_resolution}")
            report_lines.append(f"  Codec: {detected_format}")
            report_lines.append(f"  Duration: {consistency.get('duration', 0):.1f}s")

            if not consistency["consistent"]:
                for issue in consistency["issues"]:
                    report_lines.append(f"  ISSUE: {issue}")

                if strict_validation:
                    raise ValueError(
                        "Video consistency check FAILED (strict mode):\n"
                        + "\n".join(consistency["issues"])
                        + "\n\nTo ignore these issues, set strict_validation=False."
                    )
                else:
                    report_lines.append(
                        "  (strict_validation=False, continuing despite issues)"
                    )

            # Check expected FPS if provided
            if expected_fps > 0.0 and detected_fps > 0.0:
                if abs(expected_fps - detected_fps) > 0.5:
                    msg = (
                        f"Expected FPS {expected_fps} but detected {detected_fps}. "
                        "Downstream nodes will use the detected FPS."
                    )
                    report_lines.append(f"  WARNING: {msg}")
                    if strict_validation:
                        raise ValueError(
                            f"FPS mismatch: expected {expected_fps}, got {detected_fps}. "
                            "Set strict_validation=False to ignore, or correct expected_fps."
                        )

            # Check if transcode is needed
            if detected_format and detector.needs_transcode(detected_format):
                report_lines.append(
                    f"  NOTE: Codec '{detected_format}' will be transcoded to H.264 "
                    "during frame extraction."
                )

        elif structure_type == "easyvolcap":
            report_lines.append("\nEasyVolcap directory structure detected.")
            report_lines.append(f"  Camera directories: {len(contents['image_dirs'])}")

            # Try to detect FPS and resolution from existing images
            if contents["image_dirs"]:
                first_cam = Path(contents["image_dirs"][0])
                image_files = sorted(
                    [f for f in first_cam.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS]
                )
                if image_files:
                    try:
                        from PIL import Image
                        with Image.open(str(image_files[0])) as img:
                            w, h = img.size
                            detected_resolution = f"{w}x{h}"
                            report_lines.append(f"  Resolution: {detected_resolution}")
                    except Exception:
                        pass
                    report_lines.append(f"  Frames per camera: {len(image_files)}")

        # ── Step 4: Build DATASET_INFO ───────────────────────────────────
        dataset_info = create_empty_dataset_info()

        # Resolve dataset root using EnvManager
        dataset_root = self.env.get_dataset_root(dataset_name)
        sentinel_dir = self.env.get_sentinel_dir(dataset_name)

        dataset_info.update({
            "dataset_name": dataset_name,
            "dataset_root": dataset_root,
            "video_folder": video_folder,
            "video_paths": video_paths,
            "camera_count": actual_camera_count,
            "detected_fps": detected_fps,
            "detected_resolution": detected_resolution,
            "detected_format": detected_format,
            "has_calibration": has_calibration,
            "has_masks": has_masks,
            "has_vhulls": contents["has_vhulls"],
            "structure_type": structure_type,
            "image_dirs": contents.get("image_dirs", []),
            "logs_dir": self.env.get_log_dir(),
            "sentinel_dir": sentinel_dir,
            "env_info": self.env.get_env_info_dict(),
            "warnings": contents.get("warnings", []),
        })

        # ── Step 5: Create dataset directory structure ───────────────────
        report_lines.append(f"\nCreating dataset structure at: {dataset_root}")
        dir_info = create_dataset_dirs(dataset_root, actual_camera_count)
        report_lines.append(f"  Created {len(dir_info['camera_dirs'])} camera directories")
        report_lines.append(f"  Created {len(dir_info['mask_dirs'])} mask directories")

        # Copy calibration files if they exist in the source folder
        if has_calibration:
            self._copy_calibration_files(video_folder, dataset_root)
            report_lines.append("  Copied calibration files (extri.yml, intri.yml)")

        # ── Step 6: Build final report ───────────────────────────────────
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("FOLDER INGEST COMPLETE")
        report_lines.append("=" * 60)
        report_lines.append(f"  Dataset name:    {dataset_name}")
        report_lines.append(f"  Dataset root:    {dataset_root}")
        report_lines.append(f"  Camera count:    {actual_camera_count}")
        report_lines.append(f"  Structure type:  {structure_type}")
        report_lines.append(f"  Has calibration: {has_calibration}")
        report_lines.append(f"  Has masks:       {has_masks}")
        if detected_fps > 0:
            report_lines.append(f"  Detected FPS:    {detected_fps}")
        if detected_resolution:
            report_lines.append(f"  Resolution:      {detected_resolution}")
        report_lines.append("=" * 60)

        validation_report = "\n".join(report_lines)
        self._node_logger.info(validation_report)

        return (dataset_info, has_calibration, has_masks, validation_report)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _copy_calibration_files(self, source_dir: str, dest_dir: str) -> None:
        """Copy extri.yml and intri.yml from source to dataset root."""
        import shutil

        src = Path(source_dir)
        dst = Path(dest_dir)

        for fname in ("extri.yml", "intri.yml"):
            src_file = src / fname
            dst_file = dst / fname
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(str(src_file), str(dst_file))
                self._node_logger.info(f"Copied {fname} to {dest_dir}")

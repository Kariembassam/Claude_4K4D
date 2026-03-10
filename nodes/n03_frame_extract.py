"""
ComfyUI-4K4D Node 3: Frame Extract
====================================
Extracts frames from videos, auto-detects format via ffprobe,
handles sync alignment, and creates the images/ directory structure.
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import CATEGORIES, DATASET_INFO_TYPE, DEFAULTS, FRAME_NAME_FORMAT
from ..core.format_detector import FormatDetector
from ..core.sync_aligner import SyncAligner
from ..core.checkpoint_manager import CheckpointManager
from ..core.dataset_structure import create_dataset_dirs

logger = logging.getLogger("4K4D.n03_frame_extract")


class FourK4D_FrameExtract(BaseEasyVolcapNode):
    """
    Extracts frames from multi-camera video files.

    Handles:
    - Auto-detection of video format via ffprobe
    - Transcoding to H.264 baseline if needed
    - Audio waveform cross-correlation sync alignment
    - Frame extraction to EasyVolcap directory structure
    - Resolution scaling for faster training
    """

    CATEGORY = CATEGORIES["preprocessing"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("dataset_info", "frame_count", "sync_offsets", "preview_grid")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "resolution_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1
                }),
                "fps_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "frame_end": ("INT", {"default": -1, "min": -1, "max": 99999}),
                "sync_method": (["auto", "audio_xcorr", "timecode", "none"], {"default": "auto"}),
                "sync_max_offset_frames": ("INT", {"default": 10, "min": 0, "max": 100}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, resolution_scale=0.5, fps_override=0.0,
                frame_start=0, frame_end=-1, sync_method="auto",
                sync_max_offset_frames=10, unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, resolution_scale, fps_override,
            frame_start, frame_end, sync_method, sync_max_offset_frames, unique_id
        )

    def _run(self, dataset_info, resolution_scale, fps_override,
             frame_start, frame_end, sync_method, sync_max_offset_frames, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root", "dataset_name"])

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info["dataset_name"]
        camera_count = dataset_info.get("camera_count", 0)

        # Check checkpoint
        cm = CheckpointManager(os.path.join(dataset_root, ".sentinels"))
        if cm.is_completed("frame_extract"):
            meta = cm.get_metadata("frame_extract")
            self._node_logger.info("Frame extraction already completed, skipping")
            return (
                self._update_dataset_info(dataset_info, {
                    "sequence_length": meta.get("frame_count", 0),
                }),
                meta.get("frame_count", 0),
                json.dumps(meta.get("sync_offsets", {})),
                self._create_error_image("Cached — frames already extracted"),
            )

        cm.mark_started("frame_extract")

        runner = self._create_runner()
        detector = FormatDetector()
        video_paths = dataset_info.get("video_paths", [])

        # If we have videos, extract frames
        frame_count = 0
        sync_offsets_data = {}

        if video_paths and len(video_paths) > 0:
            # Detect format of first video
            video_info = detector.detect_video_info(video_paths[0])
            fps = fps_override if fps_override > 0 else video_info.get("fps", 30.0)

            # Sync alignment
            if sync_method != "none" and len(video_paths) > 1:
                aligner = SyncAligner()
                sync_result = aligner.align_videos(
                    video_paths, method=sync_method,
                    max_offset_frames=sync_max_offset_frames, fps=fps,
                )
                sync_offsets_data = sync_result
            else:
                sync_offsets_data = {
                    "offsets": {f"{i:02d}": 0 for i in range(len(video_paths))},
                    "method_used": "none",
                    "confidence": 1.0,
                    "warnings": [],
                }

            # Create directory structure
            dirs = create_dataset_dirs(dataset_root, len(video_paths))

            # Extract frames from each video
            for cam_idx, video_path in enumerate(video_paths):
                cam_id = f"{cam_idx:02d}"
                output_dir = os.path.join(dataset_root, "images", cam_id)
                offset = sync_offsets_data.get("offsets", {}).get(cam_id, 0)

                # Build ffmpeg command
                cmd = ["ffmpeg", "-y", "-i", str(video_path)]

                # Apply sync offset
                if offset > 0:
                    cmd.extend(["-ss", str(offset / fps)])

                # Apply frame range
                if frame_start > 0:
                    cmd.extend(["-ss", str(frame_start / fps)])
                if frame_end > 0:
                    cmd.extend(["-t", str((frame_end - frame_start) / fps)])

                # Resolution scaling
                if resolution_scale < 1.0:
                    scale_w = int(video_info.get("width", 1920) * resolution_scale)
                    scale_h = int(video_info.get("height", 1080) * resolution_scale)
                    # Ensure even dimensions
                    scale_w = scale_w - (scale_w % 2)
                    scale_h = scale_h - (scale_h % 2)
                    cmd.extend(["-vf", f"scale={scale_w}:{scale_h}"])

                cmd.extend([
                    "-qscale:v", "2",
                    os.path.join(output_dir, "%06d.jpg"),
                ])

                result = runner.run(
                    cmd, progress_parser=None, unique_id=unique_id, timeout_seconds=3600,
                )

                if not result.success:
                    self._node_logger.error(f"Frame extraction failed for camera {cam_id}")

            # Count extracted frames
            first_cam_dir = os.path.join(dataset_root, "images", "00")
            if os.path.exists(first_cam_dir):
                frame_count = len([
                    f for f in os.listdir(first_cam_dir)
                    if f.endswith(('.jpg', '.png'))
                ])

        else:
            # Already have images, count them
            images_dir = os.path.join(dataset_root, "images")
            if os.path.exists(images_dir):
                cam_dirs = sorted([
                    d for d in os.listdir(images_dir)
                    if os.path.isdir(os.path.join(images_dir, d))
                ])
                if cam_dirs:
                    first_cam = os.path.join(images_dir, cam_dirs[0])
                    frame_count = len([
                        f for f in os.listdir(first_cam)
                        if f.endswith(('.jpg', '.png'))
                    ])

        # Generate preview grid
        preview = self._generate_preview_grid(dataset_root, camera_count or len(video_paths))

        # Mark completed
        cm.mark_completed("frame_extract", {
            "frame_count": frame_count,
            "sync_offsets": sync_offsets_data,
        })

        updated_info = self._update_dataset_info(dataset_info, {
            "sequence_length": frame_count,
        })

        return (
            updated_info,
            frame_count,
            json.dumps(sync_offsets_data, indent=2),
            preview,
        )

    def _generate_preview_grid(self, dataset_root, camera_count):
        """Generate a tiled preview of frame 0 from all cameras."""
        try:
            import numpy as np
            from PIL import Image

            images = []
            images_dir = os.path.join(dataset_root, "images")

            for cam_idx in range(min(camera_count, 10)):
                cam_dir = os.path.join(images_dir, f"{cam_idx:02d}")
                frame_path = os.path.join(cam_dir, "000000.jpg")
                if os.path.exists(frame_path):
                    img = Image.open(frame_path).convert("RGB")
                    img = img.resize((320, 180))
                    images.append(np.array(img))

            if not images:
                return self._create_error_image("No frames extracted yet")

            # Tile images in a grid
            cols = min(5, len(images))
            rows = (len(images) + cols - 1) // cols
            tile_h, tile_w = 180, 320

            grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
            for i, img in enumerate(images):
                r, c = divmod(i, cols)
                grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = img

            # Convert to ComfyUI IMAGE format [B, H, W, C] float32
            return grid[None, :, :, :].astype(np.float32) / 255.0

        except Exception as e:
            self._node_logger.warning(f"Failed to generate preview: {e}")
            return self._create_error_image("Preview generation failed")

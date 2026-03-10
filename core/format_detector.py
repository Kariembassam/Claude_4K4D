"""
ComfyUI-4K4D Format Detector
=============================
Detects input data format using ffprobe. Identifies video codec, resolution,
FPS, and validates that input files are suitable for the 4K4D pipeline.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .constants import SUPPORTED_VIDEO_FORMATS, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger("4K4D.format_detector")


class FormatDetector:
    """
    Detects and validates input video/image formats using ffprobe.

    Handles:
    - Video format detection (codec, resolution, fps, duration)
    - Image format detection
    - Folder structure validation
    - Auto-detection of EasyVolcap-compatible directory layouts
    """

    def __init__(self):
        self._ffprobe_available: Optional[bool] = None

    @property
    def ffprobe_available(self) -> bool:
        """Check if ffprobe is available on the system."""
        if self._ffprobe_available is None:
            try:
                subprocess.run(
                    ["ffprobe", "-version"],
                    capture_output=True, timeout=10
                )
                self._ffprobe_available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._ffprobe_available = False
        return self._ffprobe_available

    def detect_video_info(self, video_path: str) -> dict:
        """
        Detect video file information using ffprobe.

        Returns:
            dict with keys: codec, width, height, fps, duration, format_name,
                            has_audio, audio_codec, bit_rate
        """
        info = {
            "codec": "unknown",
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "duration": 0.0,
            "format_name": "unknown",
            "has_audio": False,
            "audio_codec": "",
            "bit_rate": 0,
            "frame_count": 0,
            "valid": False,
            "error": "",
        }

        if not Path(video_path).exists():
            info["error"] = f"File not found: {video_path}"
            return info

        if not self.ffprobe_available:
            info["error"] = (
                "ffprobe not found. Install ffmpeg to detect video formats. "
                "On RunPod, run: apt-get install -y ffmpeg"
            )
            return info

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                info["error"] = f"ffprobe failed: {result.stderr[:200]}"
                return info

            data = json.loads(result.stdout)

            # Parse video stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["codec"] = stream.get("codec_name", "unknown")
                    info["width"] = int(stream.get("width", 0))
                    info["height"] = int(stream.get("height", 0))

                    # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
                    fps_str = stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        if int(den) > 0:
                            info["fps"] = round(int(num) / int(den), 3)

                    info["frame_count"] = int(stream.get("nb_frames", 0))
                    info["duration"] = float(stream.get("duration", 0))

                elif stream.get("codec_type") == "audio":
                    info["has_audio"] = True
                    info["audio_codec"] = stream.get("codec_name", "")

            # Parse format info
            fmt = data.get("format", {})
            info["format_name"] = fmt.get("format_name", "unknown")
            info["bit_rate"] = int(fmt.get("bit_rate", 0))
            if info["duration"] == 0:
                info["duration"] = float(fmt.get("duration", 0))

            info["valid"] = info["width"] > 0 and info["height"] > 0

        except json.JSONDecodeError:
            info["error"] = "Failed to parse ffprobe output"
        except subprocess.TimeoutExpired:
            info["error"] = "ffprobe timed out analyzing the video"
        except Exception as e:
            info["error"] = f"Error detecting video format: {str(e)}"

        return info

    def detect_folder_contents(self, folder_path: str) -> dict:
        """
        Analyze a folder to determine what it contains.

        Returns:
            dict with keys: video_files, image_dirs, has_calibration,
                            has_masks, has_vhulls, structure_type
        """
        folder = Path(folder_path)
        result = {
            "video_files": [],
            "image_dirs": [],
            "has_calibration": False,
            "has_masks": False,
            "has_vhulls": False,
            "structure_type": "unknown",  # "videos", "easyvolcap", "images_flat"
            "camera_count": 0,
            "warnings": [],
        }

        if not folder.exists():
            result["warnings"].append(f"Folder does not exist: {folder_path}")
            return result

        if not folder.is_dir():
            result["warnings"].append(f"Path is not a directory: {folder_path}")
            return result

        # Check for video files
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                result["video_files"].append(str(f))

        # Check for EasyVolcap image directory structure (images/00/, images/01/, ...)
        images_dir = folder / "images"
        if images_dir.is_dir():
            cam_dirs = sorted([
                d for d in images_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])
            if cam_dirs:
                result["image_dirs"] = [str(d) for d in cam_dirs]
                result["camera_count"] = len(cam_dirs)

        # Check for calibration files
        result["has_calibration"] = (
            (folder / "extri.yml").exists() and (folder / "intri.yml").exists()
        )

        # Check for masks
        masks_dir = folder / "masks"
        if masks_dir.is_dir():
            mask_subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
            result["has_masks"] = len(mask_subdirs) > 0

        # Check for visual hulls
        result["has_vhulls"] = (folder / "vhulls").is_dir()

        # Determine structure type
        if result["image_dirs"]:
            result["structure_type"] = "easyvolcap"
            if result["camera_count"] == 0:
                result["camera_count"] = len(result["image_dirs"])
        elif result["video_files"]:
            result["structure_type"] = "videos"
            result["camera_count"] = len(result["video_files"])
        else:
            # Check for flat images
            image_files = [
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
            ]
            if image_files:
                result["structure_type"] = "images_flat"
                result["warnings"].append(
                    "Found flat images without camera subdirectories. "
                    "Cannot determine camera count from flat structure."
                )

        return result

    def validate_videos_consistency(self, video_paths: list) -> dict:
        """
        Check that all video files have consistent properties.

        Returns:
            dict with keys: consistent, fps, resolution, duration, issues
        """
        if not video_paths:
            return {"consistent": False, "issues": ["No video files provided"]}

        infos = []
        for vp in video_paths:
            info = self.detect_video_info(vp)
            if not info["valid"]:
                return {
                    "consistent": False,
                    "issues": [f"Invalid video: {vp} — {info['error']}"]
                }
            infos.append(info)

        issues = []
        ref = infos[0]

        # Check resolution consistency
        for i, info in enumerate(infos[1:], 1):
            if info["width"] != ref["width"] or info["height"] != ref["height"]:
                issues.append(
                    f"Camera {i} resolution ({info['width']}x{info['height']}) "
                    f"differs from camera 0 ({ref['width']}x{ref['height']})"
                )

        # Check FPS consistency
        for i, info in enumerate(infos[1:], 1):
            if abs(info["fps"] - ref["fps"]) > 0.1:
                issues.append(
                    f"Camera {i} FPS ({info['fps']}) differs from "
                    f"camera 0 ({ref['fps']})"
                )

        # Check duration consistency (within 5%)
        for i, info in enumerate(infos[1:], 1):
            if ref["duration"] > 0:
                diff_pct = abs(info["duration"] - ref["duration"]) / ref["duration"] * 100
                if diff_pct > 5:
                    issues.append(
                        f"Camera {i} duration ({info['duration']:.1f}s) differs from "
                        f"camera 0 ({ref['duration']:.1f}s) by {diff_pct:.1f}%"
                    )

        return {
            "consistent": len(issues) == 0,
            "fps": ref["fps"],
            "resolution": f"{ref['width']}x{ref['height']}",
            "duration": ref["duration"],
            "codec": ref["codec"],
            "issues": issues,
        }

    def needs_transcode(self, codec: str) -> bool:
        """Check if a video codec needs to be transcoded to H.264."""
        return codec.lower() not in ("h264", "avc")

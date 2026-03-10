"""
ComfyUI-4K4D Sync Aligner
===========================
Multi-camera temporal synchronization using audio waveform cross-correlation.

Handles both synced and unsynced multi-camera footage by:
1. Extracting audio tracks from each video
2. Cross-correlating audio waveforms against a reference camera
3. Computing per-camera frame offsets
4. Applying offsets during frame extraction
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .subprocess_runner import SubprocessRunner
from .env_manager import EnvManager
from .constants import DEFAULTS

logger = logging.getLogger("4K4D.sync_aligner")


class SyncAligner:
    """
    Handles multi-camera temporal synchronization.

    Supports multiple sync methods:
    - audio_xcorr: Cross-correlate audio waveforms (default for unknown sync)
    - timecode: Parse embedded timecode metadata
    - none: Assume cameras are already synchronized

    Usage:
        aligner = SyncAligner()
        offsets = aligner.align_videos(video_paths, method="auto")
        # offsets = {"00": 0, "01": -2, "02": 1, ...}  (frame offsets)
    """

    def __init__(self):
        self.env = EnvManager.get_instance()
        self.runner = SubprocessRunner("SyncAligner", self.env.get_log_dir())

    def align_videos(
        self,
        video_paths: list,
        method: str = "auto",
        max_offset_frames: int = None,
        fps: float = 30.0,
    ) -> dict:
        """
        Compute synchronization offsets for a set of videos.

        Args:
            video_paths: List of video file paths (one per camera)
            method: "auto", "audio_xcorr", "timecode", or "none"
            max_offset_frames: Maximum allowed offset (default from DEFAULTS)
            fps: Video frame rate (used to convert time offsets to frames)

        Returns:
            dict with keys:
            - offsets: dict mapping camera_id -> frame offset (int)
            - method_used: str
            - confidence: float (0.0-1.0)
            - warnings: list of warning messages
        """
        if max_offset_frames is None:
            max_offset_frames = DEFAULTS["sync_max_offset_frames"]

        result = {
            "offsets": {},
            "method_used": method,
            "confidence": 1.0,
            "warnings": [],
        }

        if not video_paths:
            result["warnings"].append("No video files provided")
            return result

        if method == "none":
            # Assume perfect sync
            for i in range(len(video_paths)):
                result["offsets"][f"{i:02d}"] = 0
            result["method_used"] = "none"
            return result

        if method == "auto" or method == "audio_xcorr":
            return self._align_audio_xcorr(
                video_paths, max_offset_frames, fps
            )

        if method == "timecode":
            return self._align_timecode(video_paths, fps)

        result["warnings"].append(f"Unknown sync method: {method}, using 'none'")
        for i in range(len(video_paths)):
            result["offsets"][f"{i:02d}"] = 0
        return result

    def _align_audio_xcorr(
        self, video_paths: list, max_offset_frames: int, fps: float
    ) -> dict:
        """
        Align videos using audio waveform cross-correlation.

        Steps:
        1. Extract audio from each video as WAV
        2. Load audio samples
        3. Cross-correlate each camera against camera 00
        4. Find peak correlation → frame offset
        """
        result = {
            "offsets": {},
            "method_used": "audio_xcorr",
            "confidence": 0.0,
            "warnings": [],
        }

        # Check if videos have audio
        has_audio = self._check_audio(video_paths)
        if not any(has_audio):
            result["warnings"].append(
                "No audio tracks found in any video. Cannot perform audio sync. "
                "Assuming cameras are already synchronized."
            )
            for i in range(len(video_paths)):
                result["offsets"][f"{i:02d}"] = 0
            result["confidence"] = 0.5  # Low confidence without audio
            return result

        # Camera 00 is the reference — offset is always 0
        result["offsets"]["00"] = 0

        try:
            # Try numpy-based xcorr
            offsets, confidence = self._compute_xcorr_offsets(
                video_paths, max_offset_frames, fps
            )
            result["offsets"] = offsets
            result["confidence"] = confidence

            # Warn if any offset exceeds max
            for cam_id, offset in offsets.items():
                if abs(offset) > max_offset_frames:
                    result["warnings"].append(
                        f"Camera {cam_id} has large offset ({offset} frames). "
                        "This might indicate different takes or mismatched footage."
                    )
                    result["offsets"][cam_id] = max(
                        -max_offset_frames, min(max_offset_frames, offset)
                    )

        except Exception as e:
            logger.warning(f"Audio cross-correlation failed: {e}")
            result["warnings"].append(
                f"Audio sync failed: {str(e)}. Assuming cameras are synchronized."
            )
            for i in range(len(video_paths)):
                result["offsets"][f"{i:02d}"] = 0
            result["confidence"] = 0.3

        return result

    def _check_audio(self, video_paths: list) -> list:
        """Check which videos have audio tracks."""
        has_audio = []
        for vp in video_paths:
            result = self.runner.run_simple(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    str(vp),
                ],
                timeout=10,
            )
            has_audio.append(bool(result.stdout.strip()))
        return has_audio

    def _compute_xcorr_offsets(
        self, video_paths: list, max_offset_frames: int, fps: float
    ) -> tuple:
        """
        Compute cross-correlation offsets using numpy.

        Returns:
            (offsets_dict, average_confidence)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for audio sync")

        offsets = {"00": 0}
        confidences = []

        # Extract reference audio (camera 00)
        ref_audio = self._extract_audio_samples(video_paths[0], duration=10.0)
        if ref_audio is None:
            return {f"{i:02d}": 0 for i in range(len(video_paths))}, 0.5

        for i, vp in enumerate(video_paths[1:], 1):
            cam_id = f"{i:02d}"
            cam_audio = self._extract_audio_samples(vp, duration=10.0)

            if cam_audio is None:
                offsets[cam_id] = 0
                confidences.append(0.0)
                continue

            # Cross-correlate
            correlation = np.correlate(ref_audio, cam_audio, mode="full")
            max_offset_samples = int(max_offset_frames / fps * 16000)  # Assuming 16kHz

            # Restrict search range
            center = len(correlation) // 2
            search_start = max(0, center - max_offset_samples)
            search_end = min(len(correlation), center + max_offset_samples)
            search_region = correlation[search_start:search_end]

            # Find peak
            peak_idx = np.argmax(np.abs(search_region)) + search_start
            offset_samples = peak_idx - center
            offset_frames = int(round(offset_samples / 16000 * fps))

            # Confidence: peak height relative to mean
            peak_val = abs(correlation[peak_idx])
            mean_val = np.mean(np.abs(correlation))
            confidence = min(1.0, peak_val / (mean_val + 1e-8) / 10.0)

            offsets[cam_id] = offset_frames
            confidences.append(confidence)

        avg_confidence = sum(confidences) / max(len(confidences), 1)
        return offsets, avg_confidence

    def _extract_audio_samples(self, video_path: str, duration: float = 10.0):
        """Extract audio samples from a video as numpy array."""
        try:
            import numpy as np
            import subprocess
            import struct

            # Extract raw PCM audio
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-t", str(duration),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-f", "s16le", "pipe:1",
                ],
                capture_output=True, timeout=30,
            )

            if result.returncode != 0 or not result.stdout:
                return None

            # Convert to numpy array
            samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            # Normalize
            if np.max(np.abs(samples)) > 0:
                samples = samples / np.max(np.abs(samples))

            return samples

        except Exception as e:
            logger.warning(f"Failed to extract audio from {video_path}: {e}")
            return None

    def _align_timecode(self, video_paths: list, fps: float) -> dict:
        """Align videos using embedded timecode metadata."""
        result = {
            "offsets": {},
            "method_used": "timecode",
            "confidence": 0.0,
            "warnings": [],
        }

        # Extract timecodes
        timecodes = []
        for vp in video_paths:
            tc = self._extract_timecode(vp)
            timecodes.append(tc)

        if None in timecodes:
            result["warnings"].append(
                "Not all videos have timecode metadata. "
                "Falling back to assuming cameras are synchronized."
            )
            for i in range(len(video_paths)):
                result["offsets"][f"{i:02d}"] = 0
            result["confidence"] = 0.3
            return result

        # Compute offsets relative to camera 00
        ref_tc = timecodes[0]
        for i, tc in enumerate(timecodes):
            cam_id = f"{i:02d}"
            offset_seconds = tc - ref_tc
            result["offsets"][cam_id] = int(round(offset_seconds * fps))

        result["confidence"] = 0.95
        return result

    def _extract_timecode(self, video_path: str) -> Optional[float]:
        """Extract timecode from video as seconds since midnight."""
        result = self.runner.run_simple(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format_tags=timecode",
                "-of", "csv=p=0",
                str(video_path),
            ],
            timeout=10,
        )

        tc_str = result.stdout.strip()
        if not tc_str:
            return None

        try:
            # Parse HH:MM:SS:FF or HH:MM:SS;FF
            parts = tc_str.replace(";", ":").split(":")
            if len(parts) == 4:
                h, m, s, f = map(int, parts)
                return h * 3600 + m * 60 + s + f / 30.0  # Assume 30fps for TC
        except (ValueError, IndexError):
            pass

        return None

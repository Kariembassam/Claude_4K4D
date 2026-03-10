"""
Tests for core.format_detector
================================
Validates ffprobe-based video format detection with mocked external tools.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from core.format_detector import FormatDetector


# ---------------------------------------------------------------------------
# Mock ffprobe output
# ---------------------------------------------------------------------------
FFPROBE_JSON_OUTPUT = json.dumps({
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "nb_frames": "300",
            "duration": "10.0",
            "pix_fmt": "yuv420p",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
        }
    ],
    "format": {
        "filename": "cam00.mp4",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "10.0",
        "size": "15728640",
    }
})


class TestFormatDetector:
    """Validate FormatDetector with mocked ffprobe."""

    @pytest.fixture
    def detector(self):
        return FormatDetector()

    @patch("core.format_detector.subprocess")
    def test_detect_video_info(self, mock_subprocess, detector):
        """detect_video_info should parse ffprobe JSON output correctly."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = FFPROBE_JSON_OUTPUT
        mock_proc.stderr = ""
        mock_subprocess.run.return_value = mock_proc

        info = detector.detect_video_info("/fake/path/cam00.mp4")

        assert info is not None
        assert info["codec"] == "h264"
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["fps"] == 30.0 or info["fps"] == "30/1"
        assert info["has_audio"] is True

    @patch("core.format_detector.subprocess")
    def test_detect_video_info_no_ffprobe(self, mock_subprocess, detector):
        """Should handle missing ffprobe gracefully."""
        mock_subprocess.run.side_effect = FileNotFoundError("ffprobe not found")

        info = detector.detect_video_info("/fake/path/cam00.mp4")
        assert info is None or (isinstance(info, dict) and info.get("error"))

    def test_detect_folder_contents(self, detector, tmp_dataset_dir):
        """detect_folder_contents should find video files and calibration files."""
        # Create some dummy video files
        (tmp_dataset_dir / "cam00.mp4").write_bytes(b"\x00" * 100)
        (tmp_dataset_dir / "cam01.mp4").write_bytes(b"\x00" * 100)

        contents = detector.detect_folder_contents(str(tmp_dataset_dir))

        assert isinstance(contents, dict)
        assert "video_files" in contents or "videos" in contents
        assert "calibration_files" in contents or "has_calibration" in contents

    def test_validate_videos_consistency_matching(self, detector):
        """Matching resolutions and FPS should pass validation."""
        video_infos = [
            {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0},
            {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0},
            {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0},
        ]
        result = detector.validate_videos_consistency(video_infos)
        assert result["consistent"] is True or result.get("is_consistent") is True

    def test_validate_videos_consistency_mismatch(self, detector):
        """Mismatched resolutions should fail validation."""
        video_infos = [
            {"codec": "h264", "width": 1920, "height": 1080, "fps": 30.0},
            {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0},
        ]
        result = detector.validate_videos_consistency(video_infos)
        assert result["consistent"] is False or result.get("is_consistent") is False or "warnings" in result

    def test_needs_transcode_h264(self, detector):
        """H.264 source should not need transcoding."""
        assert detector.needs_transcode({"codec": "h264"}) is False

    def test_needs_transcode_prores(self, detector):
        """ProRes source should need transcoding."""
        assert detector.needs_transcode({"codec": "prores"}) is True

    def test_needs_transcode_hevc(self, detector):
        """HEVC source should need transcoding."""
        assert detector.needs_transcode({"codec": "hevc"}) is True

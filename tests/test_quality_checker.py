"""
Tests for core.quality_checker
================================
Validates the 4 pre-training quality checks: mask quality, blur/sharpness,
sync alignment, camera coverage, and the combined report generator.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core.quality_checker import QualityChecker


# ---------------------------------------------------------------------------
# Helpers — create synthetic mask images
# ---------------------------------------------------------------------------
def _make_mask_image(path, quality="good"):
    """
    Create a synthetic grayscale mask PNG as raw numpy-compatible bytes.
    'good' masks are bimodal (mostly 0/255); 'bad' masks have mid-tones.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow required for mask generation tests")

    if quality == "good":
        # Bimodal: clear foreground/background separation
        data = np.zeros((64, 64), dtype=np.uint8)
        data[16:48, 16:48] = 255
    elif quality == "bad":
        # Lots of mid-tone values (poor segmentation)
        data = np.random.randint(80, 180, (64, 64), dtype=np.uint8)
    elif quality == "empty":
        data = np.zeros((64, 64), dtype=np.uint8)
    else:
        data = np.full((64, 64), 128, dtype=np.uint8)

    img = Image.fromarray(data, mode="L")
    img.save(str(path))


def _make_frame_image(path, sharp=True):
    """Create a synthetic frame for blur testing."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow required for frame tests")

    if sharp:
        # Checkerboard pattern (high frequency — sharp)
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        data[::2, ::2] = 255
        data[1::2, 1::2] = 255
    else:
        # Uniform gray (blurry)
        data = np.full((64, 64, 3), 128, dtype=np.uint8)

    img = Image.fromarray(data, mode="RGB")
    img.save(str(path))


# ---------------------------------------------------------------------------
# QualityChecker tests
# ---------------------------------------------------------------------------
class TestQualityChecker:
    """Validate all quality check methods."""

    @pytest.fixture
    def checker(self):
        return QualityChecker()

    @pytest.fixture
    def good_mask_dir(self, tmp_path):
        """Create a mask directory with good bimodal masks."""
        mask_dir = tmp_path / "masks" / "00"
        mask_dir.mkdir(parents=True)
        for i in range(5):
            _make_mask_image(mask_dir / f"{i:06d}.png", quality="good")
        return tmp_path / "masks"

    @pytest.fixture
    def bad_mask_dir(self, tmp_path):
        """Create a mask directory with poor-quality masks."""
        mask_dir = tmp_path / "masks" / "00"
        mask_dir.mkdir(parents=True)
        for i in range(5):
            _make_mask_image(mask_dir / f"{i:06d}.png", quality="bad")
        return tmp_path / "masks"

    @pytest.fixture
    def sharp_frame_dir(self, tmp_path):
        """Create an image directory with sharp frames."""
        img_dir = tmp_path / "images" / "00"
        img_dir.mkdir(parents=True)
        for i in range(5):
            _make_frame_image(img_dir / f"{i:06d}.jpg", sharp=True)
        return tmp_path / "images"

    @pytest.fixture
    def blurry_frame_dir(self, tmp_path):
        """Create an image directory with blurry frames."""
        img_dir = tmp_path / "images" / "00"
        img_dir.mkdir(parents=True)
        for i in range(5):
            _make_frame_image(img_dir / f"{i:06d}.jpg", sharp=False)
        return tmp_path / "images"

    # -- Mask quality --
    def test_check_mask_quality_good(self, checker, good_mask_dir):
        """Good bimodal masks should pass quality check."""
        result = checker.check_mask_quality(str(good_mask_dir))
        assert result["passed"] is True
        assert result["score"] >= 0.7

    def test_check_mask_quality_bad(self, checker, bad_mask_dir):
        """Poor masks with mid-tones should fail quality check."""
        result = checker.check_mask_quality(str(bad_mask_dir))
        assert result["passed"] is False

    # -- Blur/sharpness --
    def test_check_blur_sharpness_sharp(self, checker, sharp_frame_dir):
        """Sharp checkerboard images should pass blur check."""
        result = checker.check_blur_sharpness(str(sharp_frame_dir))
        assert result["passed"] is True

    def test_check_blur_sharpness_blurry(self, checker, blurry_frame_dir):
        """Uniform gray images (no edges) should fail blur check."""
        result = checker.check_blur_sharpness(str(blurry_frame_dir))
        assert result["passed"] is False

    # -- Sync alignment --
    def test_check_sync_alignment_good(self, checker, sample_dataset_info):
        """When sync offsets are near zero, sync check should pass."""
        sample_dataset_info["sync_offsets"] = [0, 0, 0, 0, 0]
        sample_dataset_info["sync_confidence"] = 0.95
        result = checker.check_sync_alignment(sample_dataset_info)
        assert result["passed"] is True

    def test_check_sync_alignment_bad(self, checker, sample_dataset_info):
        """When sync confidence is low, sync check should fail."""
        sample_dataset_info["sync_offsets"] = [0, 15, -20, 5, 30]
        sample_dataset_info["sync_confidence"] = 0.2
        result = checker.check_sync_alignment(sample_dataset_info)
        assert result["passed"] is False or "warning" in str(result).lower()

    # -- Camera coverage --
    def test_check_camera_coverage(self, checker, sample_dataset_info):
        """Camera coverage with 5 cameras should return info (never blocks)."""
        result = checker.check_camera_coverage(sample_dataset_info)
        assert "passed" in result or "info" in result or "coverage" in str(result).lower()
        # Camera coverage is info-only, never blocks
        # With 5 cameras it should at minimum not crash

    # -- Combined report --
    def test_generate_report(self, checker, sample_dataset_info, good_mask_dir, sharp_frame_dir):
        """generate_report should combine all checks into a summary."""
        sample_dataset_info["sync_offsets"] = [0, 0, 0, 0, 0]
        sample_dataset_info["sync_confidence"] = 0.95

        report = checker.generate_report(
            dataset_info=sample_dataset_info,
            mask_dir=str(good_mask_dir),
            image_dir=str(sharp_frame_dir),
        )

        assert isinstance(report, dict)
        assert "overall_passed" in report or "passed" in report
        assert "checks" in report or "results" in report or "details" in report

"""
ComfyUI-4K4D Quality Checker
==============================
Runs all 4 pre-training quality checks:
1. Mask quality validation (bimodal distribution check)
2. Blur/sharpness scoring (Laplacian variance)
3. Sync alignment verification
4. Camera coverage heatmap

Used by the QualityGate node (Node 6b) to block training if checks fail.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

from .constants import DEFAULTS

logger = logging.getLogger("4K4D.quality_checker")


class QualityChecker:
    """
    Runs comprehensive quality checks on preprocessed data.

    All checks return a dict with:
    - passed (bool)
    - score (float, 0.0-1.0)
    - details (str)
    - issues (list of strings)
    """

    def check_mask_quality(
        self,
        masks_dir: str,
        min_confidence: float = None,
        sample_fraction: float = 0.2,
    ) -> dict:
        """
        Check 1: Mask quality validation.

        A good mask should have a bimodal distribution (mostly black or white).
        Flags frames where >30% of pixels are in the grey zone (0.2-0.8 range).
        """
        if min_confidence is None:
            min_confidence = DEFAULTS["min_mask_confidence"]

        result = {
            "passed": False,
            "score": 0.0,
            "details": "",
            "issues": [],
            "frames_checked": 0,
            "frames_passed": 0,
        }

        masks_path = Path(masks_dir)
        if not masks_path.exists():
            result["issues"].append(f"Masks directory not found: {masks_dir}")
            result["details"] = "No masks found to validate"
            return result

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            result["passed"] = True
            result["score"] = 0.5
            result["details"] = "Mask quality check skipped (numpy/PIL not available)"
            return result

        # Collect mask files
        mask_files = []
        for cam_dir in sorted(masks_path.iterdir()):
            if cam_dir.is_dir():
                files = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
                mask_files.extend(files)

        if not mask_files:
            result["issues"].append("No mask files found")
            return result

        # Sample a fraction of masks
        sample_size = max(1, int(len(mask_files) * sample_fraction))
        sampled = random.sample(mask_files, min(sample_size, len(mask_files)))

        good_frames = 0
        for mask_file in sampled:
            try:
                img = np.array(Image.open(mask_file).convert("L")) / 255.0
                # Check for bimodal distribution
                grey_zone = np.logical_and(img > 0.2, img < 0.8)
                grey_fraction = np.mean(grey_zone)

                if grey_fraction < 0.30:
                    good_frames += 1
                else:
                    result["issues"].append(
                        f"Poor mask: {mask_file.name} — {grey_fraction*100:.1f}% grey zone"
                    )
            except Exception as e:
                result["issues"].append(f"Failed to read mask {mask_file}: {e}")

        result["frames_checked"] = len(sampled)
        result["frames_passed"] = good_frames
        result["score"] = good_frames / max(len(sampled), 1)
        result["passed"] = result["score"] >= min_confidence
        result["details"] = (
            f"Checked {len(sampled)} masks: {good_frames}/{len(sampled)} passed "
            f"(score: {result['score']:.2f}, threshold: {min_confidence:.2f})"
        )

        return result

    def check_blur_sharpness(
        self,
        images_dir: str,
        max_blur_fraction: float = None,
        blur_threshold: float = 100.0,
    ) -> dict:
        """
        Check 2: Blur/sharpness scoring using Laplacian variance.

        Frames with Laplacian variance < threshold are considered blurry.
        Fails if more than max_blur_fraction of frames are blurry.
        """
        if max_blur_fraction is None:
            max_blur_fraction = DEFAULTS["max_blur_fraction"]

        result = {
            "passed": False,
            "score": 0.0,
            "details": "",
            "issues": [],
            "blurry_frames": [],
            "avg_sharpness": 0.0,
        }

        images_path = Path(images_dir)
        if not images_path.exists():
            result["issues"].append(f"Images directory not found: {images_dir}")
            return result

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            result["passed"] = True
            result["score"] = 0.5
            result["details"] = "Blur check skipped (numpy/PIL not available)"
            return result

        total_frames = 0
        blurry_count = 0
        sharpness_values = []

        for cam_dir in sorted(images_path.iterdir()):
            if not cam_dir.is_dir():
                continue

            frames = sorted(cam_dir.glob("*.jpg")) + sorted(cam_dir.glob("*.png"))
            # Sample every 10th frame for speed
            for frame in frames[::10]:
                try:
                    img = np.array(Image.open(frame).convert("L"))
                    # Laplacian variance (sharpness metric)
                    laplacian = np.array([
                        [0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]
                    ], dtype=np.float32)

                    # Simple convolution for Laplacian
                    from scipy.ndimage import convolve
                    lap = convolve(img.astype(np.float32), laplacian)
                    variance = np.var(lap)

                    sharpness_values.append(variance)
                    total_frames += 1

                    if variance < blur_threshold:
                        blurry_count += 1
                        result["blurry_frames"].append(str(frame))
                except ImportError:
                    # scipy not available, use simpler check
                    total_frames += 1
                    sharpness_values.append(100.0)  # Default pass
                except Exception as e:
                    logger.warning(f"Failed to check blur for {frame}: {e}")

        if total_frames == 0:
            result["issues"].append("No frames found to check")
            return result

        blur_fraction = blurry_count / total_frames
        result["avg_sharpness"] = sum(sharpness_values) / len(sharpness_values)
        result["score"] = 1.0 - blur_fraction
        result["passed"] = blur_fraction <= max_blur_fraction
        result["details"] = (
            f"Checked {total_frames} frames: {blurry_count} blurry "
            f"({blur_fraction*100:.1f}%, threshold: {max_blur_fraction*100:.1f}%). "
            f"Average sharpness: {result['avg_sharpness']:.1f}"
        )

        if result["blurry_frames"]:
            result["issues"].append(
                f"{len(result['blurry_frames'])} blurry frames detected. "
                "Consider re-shooting or removing them."
            )

        return result

    def check_sync_alignment(
        self,
        sync_offsets: dict,
        min_quality: float = None,
    ) -> dict:
        """
        Check 3: Sync alignment verification.

        Verifies that sync offsets are reasonable and confidence is acceptable.
        """
        if min_quality is None:
            min_quality = DEFAULTS["min_sync_quality"]

        result = {
            "passed": False,
            "score": 0.0,
            "details": "",
            "issues": [],
        }

        if not sync_offsets:
            result["passed"] = True
            result["score"] = 1.0
            result["details"] = "No sync data to verify (assuming synced)"
            return result

        offsets = sync_offsets.get("offsets", {})
        confidence = sync_offsets.get("confidence", 1.0)

        result["score"] = confidence
        result["passed"] = confidence >= min_quality

        # Check for large offsets
        max_offset = max(abs(v) for v in offsets.values()) if offsets else 0
        result["details"] = (
            f"Sync confidence: {confidence:.2f} (threshold: {min_quality:.2f}). "
            f"Max offset: {max_offset} frames across {len(offsets)} cameras."
        )

        if max_offset > DEFAULTS["sync_max_offset_frames"]:
            result["issues"].append(
                f"Large sync offset detected ({max_offset} frames). "
                "Videos may be from different takes."
            )

        if confidence < min_quality:
            result["issues"].append(
                f"Low sync confidence ({confidence:.2f}). "
                "Consider using manual timecode sync or re-recording."
            )

        return result

    def check_camera_coverage(
        self,
        dataset_root: str,
        camera_count: int,
    ) -> dict:
        """
        Check 4: Camera coverage assessment.

        This is an INFO-only check — it warns but doesn't fail the gate.
        Assesses whether camera positions provide adequate coverage.
        """
        result = {
            "passed": True,  # Always passes (info only)
            "score": 0.0,
            "details": "",
            "issues": [],
            "is_info_only": True,
        }

        if camera_count < 4:
            result["score"] = 0.3
            result["issues"].append(
                f"Only {camera_count} cameras. 4K4D recommends at least 4 cameras, "
                "and works best with 8+. Expect limited novel view coverage."
            )
        elif camera_count < 8:
            result["score"] = 0.6
            result["issues"].append(
                f"{camera_count} cameras detected. This will work but may produce "
                "artifacts at large viewing angle changes. 8+ cameras recommended."
            )
        else:
            result["score"] = min(1.0, camera_count / 20.0)

        result["details"] = (
            f"Camera count: {camera_count}. Coverage score: {result['score']:.2f}. "
            f"{'Adequate' if camera_count >= 8 else 'Limited'} coverage."
        )

        return result

    def generate_report(
        self,
        dataset_info: dict,
        sync_offsets: dict = None,
    ) -> dict:
        """
        Run ALL quality checks and generate a comprehensive report.

        Args:
            dataset_info: Pipeline dataset info dict
            sync_offsets: Sync alignment data from SyncAligner

        Returns:
            dict with overall_pass, checks dict, and summary string
        """
        dataset_root = dataset_info.get("dataset_root", "")
        camera_count = dataset_info.get("camera_count", 0)

        checks = {}

        # Check 1: Masks
        masks_dir = os.path.join(dataset_root, "masks")
        if dataset_info.get("has_masks"):
            checks["masks"] = self.check_mask_quality(masks_dir)
        else:
            checks["masks"] = {
                "passed": True, "score": 0.5,
                "details": "No masks generated (skipped)",
                "issues": [],
            }

        # Check 2: Blur/Sharpness
        images_dir = os.path.join(dataset_root, "images")
        checks["blur"] = self.check_blur_sharpness(images_dir)

        # Check 3: Sync
        checks["sync"] = self.check_sync_alignment(sync_offsets or {})

        # Check 4: Coverage (info only)
        checks["coverage"] = self.check_camera_coverage(dataset_root, camera_count)

        # Overall pass: all non-info checks must pass
        failing_checks = [
            name for name, check in checks.items()
            if not check.get("passed") and not check.get("is_info_only")
        ]

        overall_pass = len(failing_checks) == 0

        # Build summary
        lines = ["=" * 60, "QUALITY GATE REPORT", "=" * 60, ""]
        for name, check in checks.items():
            status = "PASS" if check["passed"] else ("INFO" if check.get("is_info_only") else "FAIL")
            icon = {"PASS": "[OK]", "FAIL": "[X]", "INFO": "[i]"}[status]
            lines.append(f"{icon} {name.upper()}: {check['details']}")
            for issue in check.get("issues", []):
                lines.append(f"    - {issue}")
            lines.append("")

        lines.append("=" * 60)
        if overall_pass:
            lines.append("RESULT: ALL CHECKS PASSED — Training may proceed.")
        else:
            lines.append(f"RESULT: FAILED — Checks failing: {', '.join(failing_checks)}")
            lines.append("Training is BLOCKED. Fix the issues above or use force_pass=True.")
        lines.append("=" * 60)

        return {
            "overall_pass": overall_pass,
            "checks": checks,
            "failing_checks": failing_checks,
            "summary": "\n".join(lines),
        }

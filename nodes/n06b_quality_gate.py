"""
ComfyUI-4K4D Node 6b: Quality Gate
====================================
MANDATORY pre-training validation. Blocks training if checks fail.
This node CANNOT be bypassed without override=True AND typing "I UNDERSTAND".

Checks:
1. Mask quality validation
2. Blur/sharpness scoring + auto-discard
3. Sync alignment verification
4. Camera coverage heatmap
"""

import logging
import os

from core.base_node import BaseEasyVolcapNode
from core.constants import CATEGORIES, DATASET_INFO_TYPE, DEFAULTS
from core.quality_checker import QualityChecker

logger = logging.getLogger("4K4D.n06b_quality_gate")


class FourK4D_QualityGate(BaseEasyVolcapNode):
    """
    MANDATORY pre-training quality gate.

    This is the single most important safety mechanism in the pipeline.
    It catches data quality issues BEFORE committing to a 24-hour training run.

    The gate BLOCKS training if any check fails, unless the user explicitly
    overrides by setting override_and_proceed=True AND typing "I UNDERSTAND"
    in the override_confirmation field.

    All downstream training nodes check dataset_info["quality_gate_passed"]
    and refuse to proceed if it is False.
    """

    CATEGORY = CATEGORIES["preprocessing"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "BOOLEAN", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "dataset_info", "gate_passed", "quality_report",
        "mask_quality_heatmap", "coverage_heatmap", "sync_report",
        "blur_report"
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "min_mask_confidence": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "max_blur_fraction": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "min_sync_quality": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "override_and_proceed": ("BOOLEAN", {"default": False}),
                "override_confirmation": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, min_mask_confidence=0.75, max_blur_fraction=0.10,
                min_sync_quality=0.85, override_and_proceed=False,
                override_confirmation="", unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, min_mask_confidence, max_blur_fraction,
            min_sync_quality, override_and_proceed, override_confirmation, unique_id
        )

    def _run(self, dataset_info, min_mask_confidence, max_blur_fraction,
             min_sync_quality, override_and_proceed, override_confirmation, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root"])

        checker = QualityChecker()
        report = checker.generate_report(dataset_info)

        gate_passed = report["overall_pass"]

        # Handle override
        if not gate_passed and override_and_proceed:
            if override_confirmation.strip() == "I UNDERSTAND":
                gate_passed = True
                report["summary"] += (
                    "\n\nOVERRIDE ACTIVE: Quality gate bypassed by user. "
                    "Training will proceed despite failing checks. "
                    "Results may be poor quality."
                )
                self._node_logger.warning("Quality gate OVERRIDDEN by user")
            else:
                report["summary"] += (
                    "\n\nOVERRIDE FAILED: To bypass the quality gate, you must "
                    'type exactly "I UNDERSTAND" in the override_confirmation field. '
                    "This ensures you acknowledge the risks of proceeding with "
                    "poor quality data."
                )

        # Send quality gate event to frontend
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("4k4d.quality_gate", {
                "passed": gate_passed,
                "report": {
                    "overall_pass": report["overall_pass"],
                    "failing_checks": report.get("failing_checks", []),
                },
                "message": "PASSED" if gate_passed else "FAILED — Training blocked",
            })
        except Exception:
            pass

        # Generate visualization images
        mask_heatmap = self._create_error_image(
            f"Mask Quality: {'PASS' if report['checks'].get('masks', {}).get('passed') else 'FAIL'}"
        )
        coverage_heatmap = self._create_error_image(
            f"Coverage: {report['checks'].get('coverage', {}).get('details', 'N/A')}"
        )
        sync_image = self._create_error_image(
            f"Sync: {'PASS' if report['checks'].get('sync', {}).get('passed') else 'FAIL'}"
        )

        blur_report_str = report["checks"].get("blur", {}).get("details", "No blur check data")
        if report["checks"].get("blur", {}).get("blurry_frames"):
            blur_count = len(report["checks"]["blur"]["blurry_frames"])
            blur_report_str += f"\nBlurry frames found: {blur_count}"

        updated_info = self._update_dataset_info(dataset_info, {
            "quality_gate_passed": gate_passed,
        })

        return (
            updated_info,
            gate_passed,
            report["summary"],
            mask_heatmap,
            coverage_heatmap,
            sync_image,
            blur_report_str,
        )

"""
ComfyUI-4K4D Node 8: SuperCharge
==================================
Converts trained model to SuperChargedR4DV for real-time rendering.
"""

import logging
import os

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import CATEGORIES, DATASET_INFO_TYPE

logger = logging.getLogger("4K4D.n08_supercharge")


class FourK4D_SuperCharge(BaseEasyVolcapNode):
    """
    Converts a trained 4K4D model to the SuperCharged format
    for real-time rendering capabilities.

    Uses charger.py from the 4K4D repository to precompute
    rendering data structures.
    """

    CATEGORY = CATEGORIES["processing"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("dataset_info", "supercharged_model_path", "conversion_complete", "precompute_time_estimate")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "sampler_type": (["SuperChargedR4DV", "SuperChargedR4DVB"], {"default": "SuperChargedR4DV"}),
                "frame_sample": ("STRING", {"default": "0,None,1"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, sampler_type="SuperChargedR4DV",
                frame_sample="0,None,1", unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, sampler_type, frame_sample, unique_id
        )

    def _run(self, dataset_info, sampler_type, frame_sample, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root", "dataset_name"])

        # Check quality gate — only block if explicitly set to False
        # If key is absent or None, QualityGate was bypassed — proceed with warning
        qg_value = dataset_info.get("quality_gate_passed")
        if qg_value is False:
            raise RuntimeError(
                "Quality Gate has not passed. Cannot proceed with SuperCharge.\n"
                "Run the QualityGate node first and ensure all checks pass,\n"
                "or set override_and_proceed=True and type 'I UNDERSTAND' to bypass."
            )
        elif qg_value is None or "quality_gate_passed" not in dataset_info:
            self._node_logger.warning(
                "Quality gate was not run (node bypassed or not connected). "
                "Proceeding with SuperCharge without quality validation."
            )

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info["dataset_name"]
        easyvolcap_root = dataset_info.get("easyvolcap_root") or ""
        runner = self._create_runner()

        # Auto-select sampler based on background mode
        bg_mode = dataset_info.get("background_mode", "foreground_only")
        if bg_mode == "full_scene":
            sampler_type = "SuperChargedR4DVB"
            self._node_logger.info("Auto-selected SuperChargedR4DVB for full-scene mode")

        experiment_name = dataset_info.get("experiment_name") or f"4k4d_{name}"
        config_path = dataset_info.get("config_path") or ""

        # Build charger command
        deps_dir = os.path.join(self.env.paths["deps"], "4K4D")
        charger_script = os.path.join(deps_dir, "scripts", "realtime4dv", "charger.py")

        if not os.path.exists(charger_script) and easyvolcap_root:
            charger_script = os.path.join(easyvolcap_root, "scripts", "realtime4dv", "charger.py")

        if os.path.exists(charger_script):
            cmd = [
                "python", charger_script,
                "--sampler", sampler_type,
                "--exp_name", experiment_name,
                "--",
            ]
            if config_path and os.path.exists(str(config_path)):
                cmd.extend(["-c", f"{config_path},configs/specs/super.yaml"])

            # Ensure data_root is passed as CLI override
            cmd.append(f"dataloader_cfg.dataset_cfg.data_root={dataset_root}")
            cmd.append(f"val_dataloader_cfg.dataset_cfg.data_root={dataset_root}")

            result = runner.run(
                cmd,
                cwd=easyvolcap_root or deps_dir,
                unique_id=unique_id,
                timeout_seconds=7200,
            )

            if result.success:
                supercharged_path = os.path.join(
                    dataset_root, "data", "trained_model", experiment_name, "supercharged"
                )
                return (
                    self._update_dataset_info(dataset_info, {
                        "supercharged_path": supercharged_path,
                    }),
                    supercharged_path,
                    True,
                    f"SuperCharge completed in {result.duration_seconds:.0f}s",
                )
            else:
                return (
                    dataset_info,
                    "",
                    False,
                    f"SuperCharge FAILED: {result.error_summary[:500]}",
                )
        else:
            self._node_logger.warning("charger.py not found — skipping SuperCharge")
            return (
                dataset_info,
                "",
                False,
                "charger.py not found. Install 4K4D dependencies first.",
            )

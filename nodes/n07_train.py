"""
ComfyUI-4K4D Node 07 -- FourK4D_Train
======================================
Trains the 4K4D neural point cloud model using EasyVolcap's evc-train command.

This is the MOST COMPLEX node in the pipeline. It:
1. Validates that quality_gate_passed is True (refuses to proceed otherwise)
2. Uses ConfigGenerator to create experiment YAML from dataset_info
3. Uses CheckpointManager for crash-recovery / resume
4. Runs evc-train via SubprocessRunner with real-time PSNR monitoring
5. Watches PSNR at iteration 100 and applies thresholds:
   - PSNR < 15 dB  =>  HALT training (something is very wrong)
   - PSNR < 20 dB  =>  WARN the user (quality may be poor)
6. Returns trained model path, metrics, and a preview render

Two training modes:
- preview_static: ~200 iterations, ~10 minutes, quick sanity check
- full_sequence:  ~1600 iterations, ~24 hours, production quality
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import (
    CATEGORIES,
    DATASET_INFO_TYPE,
    DEFAULTS,
    PSNR_WARN_THRESHOLD,
    PSNR_ERROR_THRESHOLD,
    PSNR_EXPECTED_PREVIEW,
)
from ..core.checkpoint_manager import CheckpointManager
from ..core.config_generator import ConfigGenerator
from ..core.subprocess_runner import evc_train_progress_parser, evc_psnr_parser

logger = logging.getLogger("4K4D.n07_train")


class FourK4D_Train(BaseEasyVolcapNode):
    """
    Train a 4K4D neural volumetric video model.

    Takes the fully-preprocessed dataset (images, masks, calibration, visual hulls)
    and trains the 4K4D point-based radiance field. Training produces a checkpoint
    that can be fed into the SuperCharge node for real-time conversion, or directly
    into the Render node for offline rendering.

    IMPORTANT: This node REQUIRES that quality_gate_passed is True in the incoming
    dataset_info. If the quality gate has not been passed, training will refuse to
    start. This prevents wasting GPU hours on bad data.

    Training modes:
    - preview_static: 200 iterations (~10 min on RTX 4090). Use this for a quick
      sanity check before committing to a full training run.
    - full_sequence: 1600 iterations (~24 hours on RTX 4090). Production quality.

    Resume behavior:
    - If resume_training is True and a checkpoint exists from a previous run,
      training will resume from that checkpoint automatically.
    - Checkpoint files are saved every checkpoint_every_n iterations.
    """

    CATEGORY = CATEGORIES["training"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "BOOLEAN", "FLOAT", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = (
        "dataset_info",
        "training_succeeded",
        "final_psnr",
        "preview_image",
        "model_path",
        "log_text",
    )
    OUTPUT_NODE = False

    STAGE_NAME = "training"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
                "training_mode": (["preview_static", "full_sequence"],),
            },
            "optional": {
                "max_iterations": (
                    "INT",
                    {
                        "default": 200,
                        "min": 10,
                        "max": 100000,
                        "step": 10,
                        "tooltip": (
                            "Total training iterations. Preview mode overrides to 200, "
                            "full mode defaults to 1600. Higher = better quality but longer."
                        ),
                    },
                ),
                "checkpoint_every_n": (
                    "INT",
                    {
                        "default": 100,
                        "min": 10,
                        "max": 10000,
                        "step": 10,
                        "tooltip": "Save a checkpoint every N iterations for crash recovery.",
                    },
                ),
                "resume_training": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "If True and a previous checkpoint exists, resume from it "
                            "instead of starting from scratch."
                        ),
                    },
                ),
                "view_sample_range": (
                    "STRING",
                    {
                        "default": "0,None,1",
                        "tooltip": (
                            "Camera view sampling as start,end,step. "
                            "'0,None,1' uses all cameras. '0,4,1' uses cameras 0-3."
                        ),
                    },
                ),
                "frame_sample_range": (
                    "STRING",
                    {
                        "default": "0,None,1",
                        "tooltip": (
                            "Frame sampling as start,end,step. "
                            "'0,None,1' uses all frames. '0,50,2' uses every other frame up to 50."
                        ),
                    },
                ),
                "background_model": (
                    ["none", "ngp_background"],
                    {
                        "default": "none",
                        "tooltip": (
                            "Background model type. 'none' trains foreground only (recommended "
                            "for masked data). 'ngp_background' adds a neural background model."
                        ),
                    },
                ),
                "force_sparse_view": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable sparse-view optimizations (regularization, augmentation). "
                            "Recommended when camera count < 8."
                        ),
                    },
                ),
                "yaml_config_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Raw YAML config overrides appended to the generated experiment "
                            "config. Advanced users only. Leave empty for defaults."
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def execute(self, **kwargs) -> Any:
        """Entry point. Delegates to _safe_execute for error handling."""
        return self._safe_execute(self._run, **kwargs)

    def _run(
        self,
        dataset_info: dict,
        training_mode: str,
        max_iterations: int = 200,
        checkpoint_every_n: int = 100,
        resume_training: bool = True,
        view_sample_range: str = "0,None,1",
        frame_sample_range: str = "0,None,1",
        background_model: str = "none",
        force_sparse_view: bool = True,
        yaml_config_override: str = "",
        unique_id: str = None,
    ) -> tuple:
        """
        Core training logic.

        Steps:
        1. Validate dataset_info and quality gate
        2. Determine iteration count from training_mode
        3. Generate experiment YAML config
        4. Check for existing checkpoint (resume)
        5. Build and run evc-train command
        6. Monitor PSNR and apply thresholds
        7. Return results
        """
        self._node_logger.info(f"Starting training: mode={training_mode}")

        # ── Step 1: Validate dataset_info ────────────────────────────────────
        self._validate_dataset_info(
            dataset_info,
            required_keys=[
                "dataset_name",
                "dataset_root",
                "camera_count",
            ],
        )

        # Check quality gate — only block if it was explicitly set to False
        # by the QualityGate node. If the key is absent or None, it means
        # the QualityGate node was bypassed or not connected — proceed with warning.
        qg_value = dataset_info.get("quality_gate_passed")
        if qg_value is False:
            raise ValueError(
                "QUALITY GATE FAILED: Training is blocked because the quality "
                "gate check ran and did not pass. This prevents wasting GPU hours "
                "on data that may produce poor results.\n\n"
                "How to fix:\n"
                "1. Fix the issues flagged by the QualityGate node\n"
                "2. Or set override_and_proceed=True and type 'I UNDERSTAND' "
                "in the QualityGate node to bypass\n\n"
                "Common quality issues: blurry frames, poor masks, camera sync problems."
            )
        elif qg_value is None or "quality_gate_passed" not in dataset_info:
            self._node_logger.warning(
                "Quality gate was not run (node bypassed or not connected). "
                "Proceeding without quality validation."
            )
        else:
            self._node_logger.info("Quality gate PASSED — proceeding with training.")

        dataset_name = dataset_info["dataset_name"]
        dataset_root = dataset_info["dataset_root"]
        camera_count = dataset_info.get("camera_count", 5)
        easyvolcap_root = dataset_info.get("easyvolcap_root")

        # ── Step 2: Determine iterations ─────────────────────────────────────
        if training_mode == "preview_static":
            effective_iterations = min(max_iterations, DEFAULTS["preview_iterations"])
            self._node_logger.info(
                f"Preview mode: clamping iterations to {effective_iterations}"
            )
        else:
            effective_iterations = max_iterations if max_iterations > 200 else DEFAULTS["full_iterations"]
            self._node_logger.info(
                f"Full training mode: {effective_iterations} iterations"
            )

        # ── Step 2b: Ensure masks and vhulls exist ─────────────────────────
        # The 4K4D pipeline requires masks and pre-computed vhull PLY files.
        # If they don't exist yet (e.g. user skipped mask/vhull nodes),
        # auto-generate placeholder data so training can proceed.
        self._ensure_masks_exist(dataset_root, dataset_info)
        self._ensure_vhulls_exist(dataset_root, dataset_info)

        # ── Step 3: Generate experiment config ───────────────────────────────
        experiment_name = f"4k4d_{dataset_name}_{training_mode}"
        training_params = {
            "experiment_name": experiment_name,
            "max_iterations": effective_iterations,
            "checkpoint_interval": checkpoint_every_n,
            "training_mode": training_mode,
            "focal_ratio": DEFAULTS.get("focal_ratio", 0.5),
        }

        if background_model != "none":
            training_params["background_mode"] = background_model

        try:
            config_gen = ConfigGenerator()
            exp_config_path = config_gen.generate_experiment_config(
                dataset_info, training_params
            )
            self._node_logger.info(f"Generated experiment config: {exp_config_path}")
        except Exception as e:
            self._node_logger.warning(
                f"ConfigGenerator failed ({e}), building inline config"
            )
            exp_config_path = self._build_fallback_config(
                dataset_info, training_params, experiment_name
            )

        # Apply YAML overrides if provided
        if yaml_config_override and yaml_config_override.strip():
            exp_config_path = self._apply_yaml_overrides(
                exp_config_path, yaml_config_override
            )

        # ── Step 4: Checkpoint / resume ──────────────────────────────────────
        sentinel_dir = dataset_info.get(
            "sentinel_dir",
            str(Path(dataset_root) / ".sentinels"),
        )
        ckpt_mgr = CheckpointManager(sentinel_dir)

        resume_from = None
        start_iteration = 0

        if resume_training and ckpt_mgr.should_resume(self.STAGE_NAME):
            checkpoint_data = ckpt_mgr.get_latest_checkpoint(self.STAGE_NAME)
            if checkpoint_data:
                resume_from = checkpoint_data.get("path")
                start_iteration = checkpoint_data.get("iteration", 0)
                self._node_logger.info(
                    f"Resuming from checkpoint: {resume_from} (iter {start_iteration})"
                )
        elif resume_training and ckpt_mgr.is_completed(self.STAGE_NAME):
            # Already completed previously -- check if user wants to redo
            prev_meta = ckpt_mgr.get_metadata(self.STAGE_NAME)
            prev_model = prev_meta.get("model_path", "")
            if prev_model and Path(prev_model).exists():
                self._node_logger.info(
                    f"Training already completed previously. Model at: {prev_model}"
                )
                updated_info = self._update_dataset_info(dataset_info, {
                    "model_path": prev_model,
                    "experiment_name": experiment_name,
                })
                preview_img = self._load_preview_image(dataset_root, experiment_name)
                return (
                    updated_info,
                    True,
                    prev_meta.get("final_psnr", 0.0),
                    preview_img,
                    prev_model,
                    f"Training already completed (resumed from sentinel). "
                    f"Model: {prev_model}",
                )

        ckpt_mgr.mark_started(self.STAGE_NAME, {
            "training_mode": training_mode,
            "max_iterations": effective_iterations,
            "experiment_name": experiment_name,
        })

        # ── Step 5: Build evc-train command ──────────────────────────────────
        configs_to_chain = self._build_config_chain(
            dataset_info, exp_config_path, background_model, force_sparse_view
        )

        # CLI args take highest priority over config YAML values.
        # These MUST be CLI args to ensure they override everything in the
        # config chain, including any defaults from base.yaml/r4dv.yaml.
        extra_args = {
            "exp_name": experiment_name,
            "dataloader_cfg.dataset_cfg.data_root": dataset_root,
            "val_dataloader_cfg.dataset_cfg.data_root": dataset_root,
            # CRITICAL: IBR dataset requires view_sample=[0,None,1].
            # Camera selection happens at sampler level, not dataset level.
            "dataloader_cfg.dataset_cfg.view_sample": "[0,None,1]",
            "val_dataloader_cfg.dataset_cfg.view_sample": "[0,None,1]",
            # Prevent NaN bounds from camera frustum intersection
            "dataloader_cfg.dataset_cfg.intersect_camera_bounds": "False",
            "val_dataloader_cfg.dataset_cfg.intersect_camera_bounds": "False",
            # Prevent runtime vhull computation failures with placeholder masks
            "dataloader_cfg.dataset_cfg.use_vhulls": "False",
            "val_dataloader_cfg.dataset_cfg.use_vhulls": "False",
        }
        if frame_sample_range and frame_sample_range != "0,None,1":
            extra_args["val_dataloader_cfg.dataset_cfg.frame_sample"] = (
                f"[{frame_sample_range}]"
            )
        if resume_from:
            extra_args["runner_cfg.resume"] = "True"

        cmd = self.build_evc_command("evc-train", configs_to_chain, extra_args)
        self._node_logger.info(f"Training command: {' '.join(cmd)}")

        # ── Step 6: Run training with PSNR monitoring ────────────────────────
        runner = self._create_runner()
        psnr_tracker = _PSNRTracker(
            node_logger=self._node_logger,
            checkpoint_manager=ckpt_mgr,
            stage_name=self.STAGE_NAME,
            warn_threshold=PSNR_WARN_THRESHOLD,
            error_threshold=PSNR_ERROR_THRESHOLD,
        )

        # Wrap the progress parser to also track PSNR
        def combined_parser(line: str):
            """Parse both progress and PSNR from each output line."""
            psnr_val = evc_psnr_parser(line)
            if psnr_val is not None:
                # Try to extract iteration number from the same line or context
                iter_match = re.search(r'[Ii]ter[:\s]+(\d+)', line)
                current_iter = int(iter_match.group(1)) if iter_match else None
                psnr_tracker.record(psnr_val, current_iter)

            return evc_train_progress_parser(line)

        # Set timeout based on mode
        if training_mode == "preview_static":
            timeout_sec = 3600  # 1 hour max for preview
        else:
            timeout_sec = 172800  # 48 hours max for full training

        # evc-train must run from the 4K4D repo directory (not EasyVolcap)
        # because configs/exps/4k4d/*.yaml live in the 4K4D repo.
        fourk4d_root = ""
        if easyvolcap_root:
            candidate = str(Path(easyvolcap_root).parent / "4K4D")
            if Path(candidate).is_dir():
                fourk4d_root = candidate
        cwd = fourk4d_root if fourk4d_root else (easyvolcap_root if easyvolcap_root and Path(easyvolcap_root).is_dir() else None)

        # Headless rendering: EasyVolcap's R4DV model uses OpenGL for
        # point rasterization. On headless servers (RunPod, etc.) we need
        # EGL as the OpenGL platform backend. Also ensure cuda-python is
        # importable for CUDA-GL interop in gl_utils.py.
        train_env = {
            "PYOPENGL_PLATFORM": "egl",
        }

        result = runner.run(
            cmd=cmd,
            cwd=cwd,
            env=train_env,
            progress_parser=combined_parser,
            timeout_seconds=timeout_sec,
            unique_id=unique_id,
        )

        # ── Step 7: Process results ──────────────────────────────────────────
        final_psnr = psnr_tracker.latest_psnr
        training_succeeded = result.success
        log_text_parts = []

        if result.success:
            log_text_parts.append(
                f"Training completed successfully in {result.duration_seconds:.0f}s "
                f"({result.duration_seconds / 60:.1f} min)."
            )
        else:
            log_text_parts.append(f"Training FAILED: {result.error_summary}")

        # Check PSNR quality
        psnr_warnings = psnr_tracker.get_warnings()
        if psnr_warnings:
            log_text_parts.extend(psnr_warnings)

        if psnr_tracker.was_halted:
            training_succeeded = False
            log_text_parts.append(
                "TRAINING HALTED: PSNR was critically low, indicating the model "
                "is not learning. Check your input data quality and calibration."
            )

        # Locate the trained model
        model_path = self._find_model_path(dataset_root, experiment_name, easyvolcap_root)
        if model_path:
            log_text_parts.append(f"Model saved to: {model_path}")
        else:
            log_text_parts.append("WARNING: Could not locate trained model file.")
            if training_succeeded:
                log_text_parts.append(
                    "Training reported success but model file not found. "
                    "Check the log file for details."
                )

        # Save checkpoint info
        if training_succeeded and model_path:
            ckpt_mgr.mark_completed(self.STAGE_NAME, {
                "model_path": model_path,
                "final_psnr": final_psnr,
                "iterations": effective_iterations,
                "training_mode": training_mode,
                "duration_seconds": result.duration_seconds,
            })
            ckpt_mgr.save_checkpoint(self.STAGE_NAME, {
                "path": model_path,
                "iteration": effective_iterations,
                "psnr": final_psnr,
            })
        else:
            ckpt_mgr.mark_failed(
                self.STAGE_NAME,
                result.error_summary or "Unknown failure",
            )

        # Load preview image if available
        preview_img = self._load_preview_image(dataset_root, experiment_name)

        # Build log text
        log_text_parts.append(f"\nFinal PSNR: {final_psnr:.2f} dB")
        log_text_parts.append(f"Log file: {result.log_path}")
        log_text = "\n".join(log_text_parts)

        # Update dataset_info
        updates = {
            "model_path": model_path or "",
            "experiment_name": experiment_name,
        }
        if psnr_tracker.get_warnings():
            updates.setdefault("warnings", list(dataset_info.get("warnings", [])))
            updates["warnings"].extend(psnr_tracker.get_warnings())

        updated_info = self._update_dataset_info(dataset_info, updates)

        self._node_logger.info(
            f"Training finished. success={training_succeeded}, psnr={final_psnr:.2f}"
        )

        return (
            updated_info,
            training_succeeded,
            final_psnr,
            preview_img,
            model_path or "",
            log_text,
        )

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _build_config_chain(
        self,
        dataset_info: dict,
        exp_config_path: str,
        background_model: str,
        force_sparse_view: bool,
    ) -> list:
        """
        Build the config chain for evc-train.

        EasyVolcap chains configs with commas: base,dataset,experiment.

        IMPORTANT: We use base.yaml + r4dv.yaml as the model architecture base,
        NOT the renbody-specific 4k4d_0013_01_r4.yaml which pulls in dataset-
        specific configs (ratio.yaml, 0013_01_obj.yaml) that conflict with
        custom datasets. Our experiment YAML provides all dataset settings.
        """
        configs = []

        # Generic model architecture configs (no dataset-specific settings)
        configs.append("configs/base.yaml")
        configs.append("configs/models/r4dv.yaml")

        # Experiment config (generated above — provides ALL dataset settings)
        configs.append(exp_config_path)

        # Background model config
        if background_model == "ngp_background":
            bg_cfg = "configs/exps/4k4d/4k4d_bg_ngp.yaml"
            evc_root = dataset_info.get("easyvolcap_root", "")
            if evc_root and Path(evc_root, bg_cfg).exists():
                configs.append(bg_cfg)

        return configs

    def _ensure_masks_exist(self, dataset_root: str, dataset_info: dict) -> None:
        """
        Ensure mask images exist for all cameras and frames.

        If masks are missing, create white (all-foreground) placeholder masks.
        These are required by the r4dv model even when use_masks=False in config,
        because mask.yaml may still be loaded via parent configs.
        """
        try:
            from PIL import Image
        except ImportError:
            self._node_logger.warning("PIL not available, skipping mask auto-generation")
            return

        images_dir = Path(dataset_root) / "images"
        mask_dir = Path(dataset_root) / "mask"

        if not images_dir.exists():
            return

        camera_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        if not camera_dirs:
            return

        masks_needed = False
        for cam_dir in camera_dirs:
            cam_mask_dir = mask_dir / cam_dir.name
            if not cam_mask_dir.exists():
                masks_needed = True
                break
            # Check if at least one mask file exists
            mask_files = list(cam_mask_dir.glob("*.png"))
            if not mask_files:
                masks_needed = True
                break

        if not masks_needed:
            self._node_logger.info("Masks already exist, skipping auto-generation")
            return

        self._node_logger.info("Auto-generating white placeholder masks...")
        total_created = 0

        for cam_dir in camera_dirs:
            cam_mask_dir = mask_dir / cam_dir.name
            cam_mask_dir.mkdir(parents=True, exist_ok=True)

            # Get image files to determine resolution and count
            img_files = sorted(
                [f for f in cam_dir.iterdir()
                 if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            )

            for img_file in img_files:
                mask_file = cam_mask_dir / f"{img_file.stem}.png"
                if mask_file.exists():
                    continue

                # Read image to get dimensions
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                except Exception:
                    w, h = 960, 540  # Fallback resolution

                # Create white mask (all foreground)
                mask_img = Image.new('L', (w, h), 255)
                mask_img.save(str(mask_file))
                total_created += 1

        if total_created > 0:
            self._node_logger.info(f"Created {total_created} white placeholder masks")

    def _ensure_vhulls_exist(self, dataset_root: str, dataset_info: dict) -> None:
        """
        Ensure pre-computed PLY point clouds exist in the vhulls/ directory.

        The 4K4D PointPlanesSampler requires PLY files to initialize the
        4D Gaussian representation. If they don't exist, generate synthetic
        point clouds with random points within the scene bounds.
        """
        import struct

        vhulls_dir = Path(dataset_root) / "vhulls"
        images_dir = Path(dataset_root) / "images"

        if not images_dir.exists():
            return

        # Determine frame count from first camera directory
        camera_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        if not camera_dirs:
            return

        frame_count = len(list(camera_dirs[0].glob("*.jpg"))) + len(list(camera_dirs[0].glob("*.png")))
        if frame_count == 0:
            return

        # Check if vhulls already exist
        if vhulls_dir.exists():
            existing_plys = list(vhulls_dir.glob("*.ply"))
            if len(existing_plys) >= frame_count:
                self._node_logger.info(
                    f"Vhulls already exist ({len(existing_plys)} PLY files), skipping"
                )
                return

        self._node_logger.info(
            f"Auto-generating {frame_count} synthetic vhull PLY files..."
        )
        vhulls_dir.mkdir(parents=True, exist_ok=True)

        # Parse bounds from dataset_info
        bounds_str = dataset_info.get("bounds", "[[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]")
        try:
            import json
            bounds = json.loads(bounds_str.replace("'", '"'))
            bmin = [float(x) for x in bounds[0]]
            bmax = [float(x) for x in bounds[1]]
        except Exception:
            bmin, bmax = [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]

        # Scale bounds inward to keep points in center
        margin = 0.2
        bmin_inner = [b + (bmax[i] - b) * margin for i, b in enumerate(bmin)]
        bmax_inner = [b - (b - bmin[i]) * margin for i, b in enumerate(bmax)]

        num_points = 5000

        import random
        random.seed(42)

        for frame_idx in range(frame_count):
            ply_path = vhulls_dir / f"{frame_idx:06d}.ply"
            if ply_path.exists():
                continue

            # Generate random points within bounds
            points = []
            for _ in range(num_points):
                x = random.uniform(bmin_inner[0], bmax_inner[0])
                y = random.uniform(bmin_inner[1], bmax_inner[1])
                z = random.uniform(bmin_inner[2], bmax_inner[2])
                r, g, b = random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)
                points.append((x, y, z, r, g, b))

            # Write binary little-endian PLY
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {num_points}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )

            with open(ply_path, 'wb') as f:
                f.write(header.encode('ascii'))
                for pt in points:
                    f.write(struct.pack('<fff', pt[0], pt[1], pt[2]))
                    f.write(struct.pack('<BBB', pt[3], pt[4], pt[5]))

        self._node_logger.info(f"Created {frame_count} synthetic vhull PLY files in {vhulls_dir}")

    def _build_fallback_config(
        self,
        dataset_info: dict,
        training_params: dict,
        experiment_name: str,
    ) -> str:
        """Build a minimal experiment config without Jinja2 templates."""
        import json

        dataset_root = dataset_info["dataset_root"]
        config_dir = Path(dataset_root)
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{experiment_name}_exp.yaml"

        seq_len = dataset_info.get("sequence_length", 1)
        lines = [
            "# Auto-generated by ComfyUI-4K4D (fallback mode)",
            f"# Experiment: {experiment_name}",
            "",
            f"exp_name: {experiment_name}",
            "",
            "runner_cfg:",
            "    epochs: 1",
            f"    ep_iter: {training_params.get('max_iterations', 200)}",
            "    save_ep: 1",
            "    save_latest_ep: 1",
            "    eval_ep: 1",
            "",
            "dataloader_cfg:",
            "    dataset_cfg:",
            f"        data_root: {dataset_root}",
            "        images_dir: images",
            "        view_sample: [0, null, 1]",
            f"        frame_sample: [0, {seq_len}, 1]",
            "        ratio: 0.5",
            "        intersect_camera_bounds: False",
            "        use_vhulls: False",
            "        bounds: [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]",
            "",
            "val_dataloader_cfg:",
            "    dataset_cfg:",
            f"        data_root: {dataset_root}",
            "        images_dir: images",
            "        view_sample: [0, null, 1]",
            "        ratio: 0.5",
            "        focal_ratio: 0.5",
            "        intersect_camera_bounds: False",
            "        use_vhulls: False",
            "        bounds: [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]",
            "",
            "model_cfg:",
            "    sampler_cfg:",
            "        bg_brightness: 0.0",
        ]

        config_path.write_text("\n".join(lines))
        self._node_logger.info(f"Wrote fallback config: {config_path}")
        return str(config_path)

    def _apply_yaml_overrides(self, config_path: str, override_text: str) -> str:
        """
        Append YAML override text to the experiment config.

        Creates a new config file that includes the original and appends overrides.
        """
        override_path = Path(config_path).with_suffix(".override.yaml")

        original_content = Path(config_path).read_text()
        combined = (
            f"{original_content}\n"
            f"\n# --- User YAML Overrides ---\n"
            f"{override_text}\n"
        )
        override_path.write_text(combined)

        self._node_logger.info(f"Applied YAML overrides to: {override_path}")
        return str(override_path)

    def _find_model_path(
        self,
        dataset_root: str,
        experiment_name: str,
        easyvolcap_root: str = None,
    ) -> str:
        """
        Locate the trained model checkpoint file.

        EasyVolcap saves checkpoints to data/trained_model/{exp_name}/
        as numbered .npz files (e.g. 199.npz, 1599.npz) or latest.pt.
        The save location is relative to the CWD where evc-train ran,
        which is the 4K4D repo directory.
        """
        search_dirs = []

        # evc-train runs from the 4K4D repo dir — check there first
        if easyvolcap_root:
            fourk4d_root = Path(easyvolcap_root).parent / "4K4D"
            if fourk4d_root.is_dir():
                search_dirs.extend([
                    fourk4d_root / "data" / "trained_model" / experiment_name,
                    fourk4d_root / "data" / "record" / experiment_name,
                    fourk4d_root / "data" / "result" / experiment_name,
                ])

        # Common EasyVolcap output locations
        if easyvolcap_root:
            search_dirs.extend([
                Path(easyvolcap_root) / "data" / "trained_model" / experiment_name,
                Path(easyvolcap_root) / "data" / "record" / experiment_name,
                Path(easyvolcap_root) / "data" / "result" / experiment_name,
            ])

        # Also check within the dataset root
        search_dirs.extend([
            Path(dataset_root) / "trained_model" / experiment_name,
            Path(dataset_root) / "output" / experiment_name,
        ])

        # Search for checkpoint files — EasyVolcap uses numbered .npz files
        # (e.g. 199.npz for iteration 199) and latest.pt for the full model
        checkpoint_names = [
            "latest.npz", "latest.pt", "latest.pth",
            "model.npz", "model.pt", "model.pth",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Check named checkpoints
            for ckpt_name in checkpoint_names:
                candidate = search_dir / ckpt_name
                if candidate.exists():
                    self._node_logger.info(f"Found model at: {candidate}")
                    return str(candidate)

            # Check for numbered .npz/.pt files (EasyVolcap iteration saves)
            numbered_files = sorted(
                [f for f in search_dir.iterdir()
                 if f.is_file() and f.suffix in (".npz", ".pt", ".pth")
                 and f.stem.replace("-", "").isdigit()],
                key=lambda f: int(f.stem.replace("-", "")),
                reverse=True,  # highest iteration first
            )
            if numbered_files:
                self._node_logger.info(f"Found model at: {numbered_files[0]}")
                return str(numbered_files[0])

            # Also search recursively one level deep
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    for ckpt_name in checkpoint_names:
                        candidate = subdir / ckpt_name
                        if candidate.exists():
                            self._node_logger.info(f"Found model at: {candidate}")
                            return str(candidate)

        self._node_logger.warning(
            f"Could not find trained model in any of: {[str(d) for d in search_dirs]}"
        )
        return ""

    def _load_preview_image(self, dataset_root: str, experiment_name: str):
        """
        Load a preview render image from the training output.

        EasyVolcap periodically saves evaluation renders during training.
        We look for the latest one to show as a preview.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return self._create_error_image("Preview unavailable (PIL not installed)")

        # Search for evaluation renders
        search_patterns = [
            Path(dataset_root) / "output" / experiment_name / "render" / "*.jpg",
            Path(dataset_root) / "output" / experiment_name / "render" / "*.png",
        ]

        evc_root = self.env.paths.get("node_root", "")
        if evc_root:
            search_patterns.extend([
                Path(evc_root).parent.parent / "data" / "result" / experiment_name / "*.jpg",
                Path(evc_root).parent.parent / "data" / "result" / experiment_name / "*.png",
            ])

        for pattern in search_patterns:
            parent = pattern.parent
            if not parent.exists():
                continue
            suffix = pattern.suffix
            files = sorted(parent.glob(f"*{suffix}"))
            if files:
                try:
                    img = np.array(Image.open(files[-1]).convert("RGB")).astype(np.float32) / 255.0
                    return self._numpy_to_comfy_image(img)
                except Exception as e:
                    self._node_logger.warning(f"Failed to load preview image: {e}")

        # Return a placeholder
        return self._create_placeholder_image("Training complete -- no preview available")

    def _create_placeholder_image(self, text: str):
        """Create a simple placeholder image with text as a PyTorch tensor."""
        try:
            import torch
            img = torch.zeros(1, 256, 512, 3, dtype=torch.float32)
            img[:, :, :, 1] = 0.15  # Subtle green tint for success
            img[:, :, :, 2] = 0.15
            return img
        except Exception:
            return self._create_error_image(text)


class _PSNRTracker:
    """
    Tracks PSNR values during training and applies quality thresholds.

    Monitors PSNR at specific iteration milestones:
    - At iter 100: PSNR < 20 => warn, PSNR < 15 => halt
    - Tracks overall trend for final reporting
    """

    def __init__(
        self,
        node_logger: logging.Logger,
        checkpoint_manager: CheckpointManager,
        stage_name: str,
        warn_threshold: float = PSNR_WARN_THRESHOLD,
        error_threshold: float = PSNR_ERROR_THRESHOLD,
    ):
        self.logger = node_logger
        self.ckpt_mgr = checkpoint_manager
        self.stage_name = stage_name
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold

        self.psnr_history: list = []  # [(iteration, psnr_value)]
        self.latest_psnr: float = 0.0
        self.was_halted: bool = False
        self._warnings: list = []
        self._iter_100_checked: bool = False

    def record(self, psnr: float, iteration: int = None) -> None:
        """Record a PSNR measurement and check thresholds."""
        self.latest_psnr = psnr
        self.psnr_history.append((iteration, psnr))
        self.logger.debug(f"PSNR at iter {iteration}: {psnr:.2f} dB")

        # Save periodic checkpoint
        if iteration is not None and iteration > 0 and iteration % 100 == 0:
            self.ckpt_mgr.save_checkpoint(self.stage_name, {
                "iteration": iteration,
                "psnr": psnr,
            })

        # Check thresholds at iteration ~100
        if (
            iteration is not None
            and 90 <= iteration <= 110
            and not self._iter_100_checked
        ):
            self._iter_100_checked = True
            self._check_iter_100_threshold(psnr, iteration)

    def _check_iter_100_threshold(self, psnr: float, iteration: int) -> None:
        """Apply PSNR thresholds at the ~100 iteration mark."""
        if psnr < self.error_threshold:
            self.was_halted = True
            msg = (
                f"CRITICAL: PSNR is {psnr:.2f} dB at iteration {iteration}, "
                f"which is below the halt threshold of {self.error_threshold:.1f} dB. "
                "The model is not learning properly. Possible causes:\n"
                "- Bad camera calibration\n"
                "- Incorrect masks (foreground/background swapped)\n"
                "- Corrupt input images\n"
                "- Mismatched frame ordering across cameras"
            )
            self.logger.error(msg)
            self._warnings.append(msg)

        elif psnr < self.warn_threshold:
            msg = (
                f"WARNING: PSNR is {psnr:.2f} dB at iteration {iteration}, "
                f"which is below the expected {self.warn_threshold:.1f} dB. "
                "Training will continue but quality may be poor. Consider:\n"
                "- Checking mask quality\n"
                "- Verifying camera calibration accuracy\n"
                "- Ensuring images are sharp and well-lit"
            )
            self.logger.warning(msg)
            self._warnings.append(msg)
        else:
            self.logger.info(
                f"PSNR check at iter {iteration}: {psnr:.2f} dB (OK, "
                f"threshold: {self.warn_threshold:.1f} dB)"
            )

    def get_warnings(self) -> list:
        """Return all accumulated warnings."""
        return list(self._warnings)

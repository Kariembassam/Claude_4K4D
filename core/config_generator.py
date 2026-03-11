"""
ComfyUI-4K4D Config Generator
===============================
Jinja2-based YAML configuration generator for EasyVolcap.

Renders dataset, experiment, and rendering configs from templates,
ensuring all required fields are populated and configs are valid YAML.
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import jinja2
except ImportError:
    jinja2 = None

try:
    import yaml
except ImportError:
    yaml = None

from .constants import DEFAULTS, CATEGORY_PREFIX
from .env_manager import EnvManager

logger = logging.getLogger("4K4D.config_generator")


class ConfigGenerator:
    """
    Renders EasyVolcap YAML configs from Jinja2 templates.

    Templates are stored in configs/templates/ and rendered with
    dataset-specific values from DATASET_INFO.

    Usage:
        gen = ConfigGenerator()
        config_path = gen.generate_dataset_config(dataset_info)
        exp_path = gen.generate_experiment_config(dataset_info, training_params)
    """

    def __init__(self, templates_dir: str = None):
        self.env = EnvManager.get_instance()
        self.templates_dir = templates_dir or self.env.paths["templates"]

        if jinja2 is None:
            raise ImportError(
                "jinja2 is required for config generation. "
                "Install it with: pip install jinja2"
            )

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            undefined=jinja2.StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate_dataset_config(
        self, dataset_info: dict, output_path: str = None
    ) -> str:
        """
        Generate a dataset YAML config for EasyVolcap.

        Args:
            dataset_info: Pipeline dataset info dict
            output_path: Where to write the config (auto-generated if None)

        Returns:
            Path to the generated config file
        """
        name = dataset_info.get("dataset_name", "unnamed")
        data_root = dataset_info.get("dataset_root", "")

        if output_path is None:
            output_path = os.path.join(data_root, f"{name}.yaml")

        context = {
            "dataset_name": name,
            "data_root": data_root,
            "images_dir": "images",
            "masks_dir": "masks" if dataset_info.get("has_masks") else "",
            "camera_count": dataset_info.get("camera_count", 5),
            "sequence_length": dataset_info.get("sequence_length", 1),
            "ratio": DEFAULTS.get("resolution_scale", 0.5),
            "view_sample": f"0,{dataset_info.get('camera_count', 5)},1",
            "frame_sample": f"0,{dataset_info.get('sequence_length', 1)},1",
            "force_sparse_view": dataset_info.get("camera_count", 5) < 8,
            "bounds": dataset_info.get("bounds", "[[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]"),
        }

        return self._render_template("dataset.yaml.j2", context, output_path)

    def generate_dataset_obj_config(
        self, dataset_info: dict, output_path: str = None
    ) -> str:
        """
        Generate the dataset object config ({name}_obj.yaml) with bounding box.
        """
        name = dataset_info.get("dataset_name", "unnamed")
        data_root = dataset_info.get("dataset_root", "")

        if output_path is None:
            output_path = os.path.join(data_root, f"{name}_obj.yaml")

        context = {
            "dataset_name": name,
            "parent_config": f"configs/datasets/{name}/{name}.yaml",
            "bounds": dataset_info.get("bounds", "[[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]"),
            "vhull_thresh": (
                DEFAULTS["vhull_thresh_sparse"]
                if dataset_info.get("camera_count", 5) < 8
                else DEFAULTS["vhull_thresh_dense"]
            ),
        }

        return self._render_template("dataset_obj.yaml.j2", context, output_path)

    def generate_experiment_config(
        self,
        dataset_info: dict,
        training_params: dict = None,
        output_path: str = None,
    ) -> str:
        """
        Generate a training experiment config.

        Args:
            dataset_info: Pipeline dataset info dict
            training_params: Override training parameters
            output_path: Where to write the config
        """
        name = dataset_info.get("dataset_name", "unnamed")
        data_root = dataset_info.get("dataset_root", "")
        params = training_params or {}

        if output_path is None:
            output_path = os.path.join(data_root, f"{name}_exp.yaml")

        context = {
            "dataset_name": name,
            "data_root": data_root,
            "experiment_name": params.get("experiment_name", f"4k4d_{name}"),
            "dataset_config": dataset_info.get("config_path", f"{name}.yaml"),
            "max_iterations": params.get("max_iterations", DEFAULTS["full_iterations"]),
            "checkpoint_interval": params.get("checkpoint_interval", DEFAULTS["checkpoint_interval"]),
            "bg_brightness": params.get("bg_brightness", DEFAULTS["bg_brightness"]),
            "background_mode": dataset_info.get("background_mode", "foreground_only"),
            "training_mode": params.get("training_mode", "full_sequence"),
            "camera_count": dataset_info.get("camera_count", 5),
            "images_dir": "images",
            "ratio": DEFAULTS.get("resolution_scale", 0.5),
            "view_sample": f"0,{dataset_info.get('camera_count', 5)},1",
            "frame_sample": f"0,{dataset_info.get('sequence_length', 1)},1",
            "focal_ratio": params.get("focal_ratio", DEFAULTS["focal_ratio"]),
            "force_sparse_view": dataset_info.get("camera_count", 5) < 8,
            "bounds": dataset_info.get("bounds", "[[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]"),
        }

        return self._render_template("experiment.yaml.j2", context, output_path)

    def generate_bg_experiment_config(
        self, dataset_info: dict, output_path: str = None
    ) -> str:
        """Generate background training experiment config."""
        name = dataset_info.get("dataset_name", "unnamed")
        data_root = dataset_info.get("dataset_root", "")

        if output_path is None:
            output_path = os.path.join(data_root, f"{name}_bg_exp.yaml")

        context = {
            "dataset_name": name,
            "data_root": data_root,
            "experiment_name": f"l3mhet_{name}_bkgd",
        }

        return self._render_template("experiment_bg.yaml.j2", context, output_path)

    def generate_fg_experiment_config(
        self, dataset_info: dict, output_path: str = None
    ) -> str:
        """Generate foreground training experiment config."""
        name = dataset_info.get("dataset_name", "unnamed")
        data_root = dataset_info.get("dataset_root", "")

        if output_path is None:
            output_path = os.path.join(data_root, f"{name}_fg_exp.yaml")

        context = {
            "dataset_name": name,
            "data_root": data_root,
            "experiment_name": f"4k4d_{name}_fg",
        }

        return self._render_template("experiment_fg.yaml.j2", context, output_path)

    def _render_template(self, template_name: str, context: dict, output_path: str) -> str:
        """
        Render a Jinja2 template and write the result.

        Args:
            template_name: Template filename in templates_dir
            context: Template variables
            output_path: Where to write the rendered config

        Returns:
            Absolute path to the generated file
        """
        try:
            template = self.jinja_env.get_template(template_name)
        except jinja2.TemplateNotFound:
            # If template doesn't exist, generate a basic config
            logger.warning(f"Template {template_name} not found, generating inline config")
            return self._generate_inline_config(context, output_path)

        rendered = template.render(**context)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)

        logger.info(f"Generated config: {output_path}")
        return str(output.resolve())

    def _generate_inline_config(self, context: dict, output_path: str) -> str:
        """Generate a basic config without a template (fallback)."""
        if yaml is None:
            # Manual YAML generation
            lines = ["# Auto-generated by ComfyUI-4K4D"]
            for key, value in context.items():
                lines.append(f"{key}: {value}")
            content = "\n".join(lines)
        else:
            content = f"# Auto-generated by ComfyUI-4K4D\n{yaml.dump(context, default_flow_style=False)}"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)

        return str(output.resolve())

    def validate_config(self, config_path: str) -> tuple:
        """
        Validate that a generated config is valid YAML.

        Returns:
            (is_valid, message) tuple
        """
        if yaml is None:
            return True, "YAML validation skipped (pyyaml not installed)"

        try:
            with open(config_path) as f:
                yaml.safe_load(f)
            return True, "Config is valid YAML"
        except yaml.YAMLError as e:
            return False, f"Invalid YAML at {config_path}: {str(e)}"
        except FileNotFoundError:
            return False, f"Config file not found: {config_path}"

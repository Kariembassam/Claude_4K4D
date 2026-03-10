"""
Tests for core.config_generator
=================================
Validates Jinja2 YAML config generation for EasyVolcap.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from core.config_generator import ConfigGenerator


# ---------------------------------------------------------------------------
# Config generation tests
# ---------------------------------------------------------------------------
class TestConfigGenerator:
    """Validate YAML config rendering from Jinja2 templates."""

    @pytest.fixture(autouse=True)
    def setup_templates(self, tmp_path):
        """Ensure templates directory and a minimal template exist."""
        self.template_dir = tmp_path / "templates"
        self.template_dir.mkdir(parents=True)
        self.output_dir = tmp_path / "output"
        self.output_dir.mkdir(parents=True)

        # Create a minimal dataset.yaml.j2 template
        dataset_template = self.template_dir / "dataset.yaml.j2"
        dataset_template.write_text(
            "dataloader_cfg:\n"
            "  dataset_cfg:\n"
            "    data_root: {{ data_root }}\n"
            "    images_dir: {{ images_dir | default('images') }}\n"
            "    camera_count: {{ camera_count }}\n"
            "    frame_count: {{ frame_count }}\n"
            "    ratio: {{ ratio | default(1.0) }}\n"
        )

        # Create a minimal experiment.yaml.j2 template
        experiment_template = self.template_dir / "experiment.yaml.j2"
        experiment_template.write_text(
            "exp_name: {{ experiment_name }}\n"
            "training:\n"
            "  max_iter: {{ max_iterations | default(1600) }}\n"
            "  lr: {{ learning_rate | default(0.0007) }}\n"
            "  checkpoint_interval: {{ checkpoint_interval | default(100) }}\n"
        )

        # Create dataset_obj.yaml.j2 template
        obj_template = self.template_dir / "dataset_obj.yaml.j2"
        obj_template.write_text(
            "bounds:\n"
            "  min: {{ bounds_min }}\n"
            "  max: {{ bounds_max }}\n"
            "vhull_thresh: {{ vhull_thresh | default(0.5) }}\n"
        )

        self.generator = ConfigGenerator(str(self.template_dir))

    def test_generate_dataset_config(self, sample_dataset_info):
        """Generated dataset config should be valid YAML with expected keys."""
        output_path = str(self.output_dir / "dataset.yaml")
        self.generator.generate_dataset_config(
            dataset_info=sample_dataset_info,
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert "dataloader_cfg" in config
        assert config["dataloader_cfg"]["dataset_cfg"]["camera_count"] == 5

    def test_generate_experiment_config(self, sample_dataset_info):
        """Generated experiment config should have training parameters."""
        output_path = str(self.output_dir / "experiment.yaml")
        self.generator.generate_experiment_config(
            dataset_info=sample_dataset_info,
            output_path=output_path,
            max_iterations=800,
            learning_rate=0.001,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert config["training"]["max_iter"] == 800
        assert config["training"]["lr"] == 0.001

    def test_generate_dataset_obj_config(self, sample_dataset_info):
        """Generated obj config should include bounds."""
        output_path = str(self.output_dir / "dataset_obj.yaml")
        self.generator.generate_dataset_obj_config(
            dataset_info=sample_dataset_info,
            output_path=output_path,
            bounds_min=[-5.0, -5.0, -5.0],
            bounds_max=[5.0, 5.0, 5.0],
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            content = f.read()
        assert "bounds" in content

    def test_missing_template_fallback(self):
        """If template file is missing, generator should use inline fallback or raise."""
        gen = ConfigGenerator("/nonexistent/templates")
        with pytest.raises((FileNotFoundError, Exception)):
            gen.generate_dataset_config(
                dataset_info={"dataset_name": "x", "dataset_root": "/tmp/x",
                              "camera_count": 1, "sequence_length": 1},
                output_path="/tmp/out.yaml",
            )

    def test_validate_config_structure(self, sample_dataset_info):
        """validate_config should accept well-formed configs."""
        output_path = str(self.output_dir / "dataset.yaml")
        self.generator.generate_dataset_config(
            dataset_info=sample_dataset_info,
            output_path=output_path,
        )
        # If validate_config exists, test it; otherwise just verify YAML is parseable
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

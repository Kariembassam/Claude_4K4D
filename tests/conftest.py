"""
ComfyUI-4K4D Test Fixtures
===========================
Shared pytest fixtures for all test modules. Provides temporary dataset
directories, sample DATASET_INFO dicts, and mocked singletons.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so `core` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import create_empty_dataset_info, DEFAULTS, DIR_IMAGES, DIR_MASKS
from core.subprocess_runner import SubprocessResult


# ---------------------------------------------------------------------------
# Fixture: temporary dataset directory with EasyVolcap directory structure
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_dataset_dir(tmp_path):
    """
    Create a temporary directory tree mimicking the EasyVolcap dataset layout:

        <root>/
            images/
                00/  (000000.jpg .. 000004.jpg)
                01/
                02/
                03/
                04/
            masks/
                00/  (000000.png .. 000004.png)
                01/
                02/
                03/
                04/
            extri.yml
            intri.yml
    """
    dataset_root = tmp_path / "test_sequence"
    dataset_root.mkdir()

    num_cameras = 5
    num_frames = 5

    for subdir in (DIR_IMAGES, DIR_MASKS):
        for cam_idx in range(num_cameras):
            cam_dir = dataset_root / subdir / f"{cam_idx:02d}"
            cam_dir.mkdir(parents=True)
            for frame_idx in range(num_frames):
                ext = ".jpg" if subdir == DIR_IMAGES else ".png"
                frame_file = cam_dir / f"{frame_idx:06d}{ext}"
                # Write a tiny valid-enough stub (1x1 pixel files are enough for
                # path-based tests; image-content tests create their own data).
                frame_file.write_bytes(b"\x00" * 8)

    # Calibration placeholders
    (dataset_root / "extri.yml").write_text("# extrinsic placeholder\n")
    (dataset_root / "intri.yml").write_text("# intrinsic placeholder\n")

    return dataset_root


# ---------------------------------------------------------------------------
# Fixture: sample DATASET_INFO dict
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_dataset_info(tmp_dataset_dir):
    """
    Return a fully-populated DATASET_INFO dict that mirrors the schema
    defined by ``create_empty_dataset_info()`` in constants.py, but with
    plausible non-empty values.
    """
    info = create_empty_dataset_info()
    info.update({
        "dataset_name": "test_sequence",
        "dataset_root": str(tmp_dataset_dir),
        "video_paths": [
            str(tmp_dataset_dir / f"cam{i:02d}.mp4") for i in range(5)
        ],
        "camera_count": 5,
        "detected_fps": 30.0,
        "detected_resolution": "1920x1080",
        "detected_format": "h264",
        "has_calibration": True,
        "has_masks": True,
        "sequence_length": 5,
        "bounds": "[[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]",
        "model_path": None,
        "supercharged_path": None,
        "render_output": None,
        "quality_gate_passed": False,
        "dep_version_mode": "stable",
        "background_mode": "foreground_only",
        "config_path": None,
        "experiment_name": "4k4d_test_sequence",
        "easyvolcap_root": None,
        "env_info": {"is_runpod": False},
        "logs_dir": str(tmp_dataset_dir / "logs"),
        "sentinel_dir": str(tmp_dataset_dir / ".sentinels"),
        "errors": [],
        "warnings": [],
    })
    return info


# ---------------------------------------------------------------------------
# Fixture: mocked EnvManager singleton
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_env_manager(tmp_path):
    """
    Return a ``MagicMock`` that stands in for ``EnvManager.get_instance()``.

    Key attributes (``paths``, ``is_runpod``, etc.) are pre-configured so
    tests can use the manager without hitting the real filesystem or GPU.
    """
    mgr = MagicMock()
    mgr.is_runpod = False
    mgr.is_serverless = False
    mgr.python_path = sys.executable
    mgr.node_pack_root = tmp_path / "ComfyUI-4K4D"
    mgr.comfyui_root = tmp_path / "ComfyUI"

    paths = {
        "node_root": str(mgr.node_pack_root),
        "data": str(tmp_path / "data"),
        "deps": str(tmp_path / "deps"),
        "logs": str(tmp_path / "logs"),
        "exports": str(tmp_path / "exports"),
        "compiled": str(tmp_path / "compiled"),
        "configs": str(tmp_path / "configs"),
        "templates": str(tmp_path / "configs" / "templates"),
        "workflows": str(tmp_path / "workflows"),
        "web": str(tmp_path / "web"),
    }
    mgr.paths = paths
    mgr.get_log_dir.return_value = paths["logs"]
    mgr.get_dataset_root.side_effect = lambda name: str(tmp_path / "data" / name)
    mgr.get_sentinel_dir.side_effect = lambda name: str(
        tmp_path / "data" / name / ".sentinels"
    )
    mgr.gpu_info = {
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_memory_gb": 24.0,
        "cuda_version": "12.1",
        "gpu_arch": "8.9",
        "gpu_count": 1,
    }
    mgr.cuda_available = True
    mgr.get_env_info_dict.return_value = {
        "is_runpod": False,
        "python_path": sys.executable,
        "node_pack_root": str(mgr.node_pack_root),
        "comfyui_root": str(mgr.comfyui_root),
        "gpu": mgr.gpu_info,
    }
    mgr.get_cuda_env_vars.return_value = {
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "CUDA_HOME": "/usr/local/cuda",
    }
    return mgr


# ---------------------------------------------------------------------------
# Fixture: factory for SubprocessResult objects
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_subprocess_result():
    """
    Factory fixture that builds ``SubprocessResult`` instances with custom
    field overrides.  Defaults describe a successful, fast execution.

    Usage in tests::

        result = mock_subprocess_result(return_code=1, stderr="oops")
    """

    def _factory(**overrides):
        defaults = {
            "return_code": 0,
            "stdout": "ok\n",
            "stderr": "",
            "duration_seconds": 0.5,
            "was_cancelled": False,
            "was_timeout": False,
            "log_path": "/tmp/test.log",
            "error_summary": "",
        }
        defaults.update(overrides)
        return SubprocessResult(**defaults)

    return _factory

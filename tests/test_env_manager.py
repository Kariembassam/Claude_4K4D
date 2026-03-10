"""
Tests for core.env_manager.EnvManager
======================================
Covers singleton behaviour, path resolution, RunPod detection,
GPU info (mocked), and CUDA environment variable generation.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from core.env_manager import EnvManager


# ---------------------------------------------------------------------------
# Ensure singleton is reset between tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the EnvManager singleton before and after each test."""
    EnvManager.reset()
    yield
    EnvManager.reset()


# ---------------------------------------------------------------------------
# test_singleton_pattern
# ---------------------------------------------------------------------------
class TestSingletonPattern:
    def test_same_instance_returned(self):
        """get_instance() must return the exact same object on repeated calls."""
        a = EnvManager.get_instance()
        b = EnvManager.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        """After reset(), the next get_instance() must return a fresh object."""
        first = EnvManager.get_instance()
        EnvManager.reset()
        second = EnvManager.get_instance()
        assert first is not second


# ---------------------------------------------------------------------------
# test_paths_are_set
# ---------------------------------------------------------------------------
class TestPathsAreSet:
    def test_paths_dict_has_required_keys(self):
        """The paths dict must contain all expected keys."""
        env = EnvManager.get_instance()
        expected_keys = {
            "node_root", "data", "deps", "logs", "exports",
            "compiled", "configs", "templates", "workflows", "web",
        }
        assert expected_keys.issubset(set(env.paths.keys()))

    def test_paths_are_strings(self):
        env = EnvManager.get_instance()
        for key, value in env.paths.items():
            assert isinstance(value, str), f"paths['{key}'] should be a string"

    def test_node_pack_root_is_absolute(self):
        env = EnvManager.get_instance()
        assert Path(env.paths["node_root"]).is_absolute()


# ---------------------------------------------------------------------------
# test_is_runpod_detection
# ---------------------------------------------------------------------------
class TestIsRunpodDetection:
    def test_not_runpod_by_default(self):
        """On a normal dev machine without RunPod env vars, is_runpod should
        be False (unless /workspace happens to exist)."""
        env = EnvManager.get_instance()
        # We cannot guarantee /workspace doesn't exist on all machines, so
        # we patch it to make the test deterministic.
        EnvManager.reset()
        with patch.object(Path, "exists", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                env2 = EnvManager()
                assert env2.is_runpod is False

    def test_runpod_detected_via_env_var(self):
        """Setting RUNPOD_POD_ID should trigger is_runpod == True."""
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "abc123"}, clear=False):
            env = EnvManager()
            assert env.is_runpod is True

    def test_runpod_detected_via_gpu_count_env(self):
        """Setting RUNPOD_GPU_COUNT should also trigger is_runpod."""
        with patch.dict(os.environ, {"RUNPOD_GPU_COUNT": "1"}, clear=False):
            env = EnvManager()
            assert env.is_runpod is True

    def test_serverless_detection(self):
        """RUNPOD_SERVERLESS_ID should set is_serverless."""
        with patch.dict(os.environ, {"RUNPOD_SERVERLESS_ID": "srv-123"}, clear=False):
            env = EnvManager()
            assert env.is_serverless is True


# ---------------------------------------------------------------------------
# test_gpu_info
# ---------------------------------------------------------------------------
class TestGpuInfo:
    def test_gpu_info_with_mocked_torch(self):
        """When torch reports a GPU, gpu_info should contain its details."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = "12.1"

        mock_props = MagicMock()
        mock_props.total_mem = 24 * (1024 ** 3)  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.get_device_capability.return_value = (8, 9)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            env = EnvManager()
            info = env.gpu_info

            assert info["cuda_available"] is True
            assert info["gpu_name"] == "NVIDIA RTX 4090"
            assert info["gpu_memory_gb"] == 24.0
            assert info["cuda_version"] == "12.1"
            assert info["gpu_arch"] == "8.9"
            assert info["gpu_count"] == 1

    def test_gpu_info_without_torch(self):
        """When torch is not importable, gpu_info should return safe defaults."""
        with patch.dict("sys.modules", {"torch": None}):
            env = EnvManager()
            # Force re-detection by clearing cached value
            env._gpu_info = None
            info = env._detect_gpu()

            assert info["cuda_available"] is False
            assert info["gpu_name"] == "N/A"
            assert info["gpu_count"] == 0

    def test_validate_gpu_fails_without_cuda(self):
        """validate_gpu should fail when cuda_available is False."""
        env = EnvManager()
        env._gpu_info = {
            "cuda_available": False,
            "gpu_name": "N/A",
            "gpu_memory_gb": 0.0,
            "cuda_version": "N/A",
            "gpu_arch": "N/A",
            "gpu_count": 0,
        }
        is_valid, msg = env.validate_gpu()
        assert is_valid is False
        assert "No CUDA" in msg

    def test_validate_gpu_fails_low_vram(self):
        """validate_gpu should fail when VRAM is below the threshold."""
        env = EnvManager()
        env._gpu_info = {
            "cuda_available": True,
            "gpu_name": "GTX 1060",
            "gpu_memory_gb": 6.0,
            "cuda_version": "11.8",
            "gpu_arch": "6.1",
            "gpu_count": 1,
        }
        is_valid, msg = env.validate_gpu(min_vram_gb=20.0)
        assert is_valid is False
        assert "6.0" in msg

    def test_validate_gpu_passes(self):
        """validate_gpu should pass for adequate hardware."""
        env = EnvManager()
        env._gpu_info = {
            "cuda_available": True,
            "gpu_name": "NVIDIA RTX 4090",
            "gpu_memory_gb": 24.0,
            "cuda_version": "12.1",
            "gpu_arch": "8.9",
            "gpu_count": 1,
        }
        is_valid, msg = env.validate_gpu(min_vram_gb=20.0)
        assert is_valid is True
        assert "GPU OK" in msg


# ---------------------------------------------------------------------------
# test_get_cuda_env_vars
# ---------------------------------------------------------------------------
class TestGetCudaEnvVars:
    def test_cuda_env_contains_arch(self):
        """Returned env must contain TORCH_CUDA_ARCH_LIST."""
        env = EnvManager()
        env._gpu_info = {
            "cuda_available": True,
            "gpu_name": "RTX 4090",
            "gpu_memory_gb": 24.0,
            "cuda_version": "12.1",
            "gpu_arch": "8.9",
            "gpu_count": 1,
        }
        cuda_env = env.get_cuda_env_vars()
        assert "TORCH_CUDA_ARCH_LIST" in cuda_env
        assert cuda_env["TORCH_CUDA_ARCH_LIST"] == "8.9"

    def test_cuda_env_inherits_os_environ(self):
        """Returned env should be a superset of os.environ."""
        env = EnvManager()
        env._gpu_info = {
            "gpu_arch": "8.6",
            "cuda_version": "11.8",
        }
        cuda_env = env.get_cuda_env_vars()
        # PATH at least should be present
        assert "PATH" in cuda_env

    def test_cuda_home_set_when_exists(self):
        """If /usr/local/cuda exists, CUDA_HOME should be set."""
        env = EnvManager()
        env._gpu_info = {
            "gpu_arch": "8.6",
            "cuda_version": "11.8",
        }
        with patch.object(Path, "exists", return_value=True):
            cuda_env = env.get_cuda_env_vars()
            assert "CUDA_HOME" in cuda_env

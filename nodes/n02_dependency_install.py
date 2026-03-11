"""
ComfyUI-4K4D Node 2: Dependency Install
========================================
Installs all dependencies, compiles CUDA extensions lazily, clones source repos.
15-step install sequence with sentinel file tracking for idempotency.
"""

import os
import sys
import logging
from pathlib import Path

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import (
    CATEGORIES, DATASET_INFO_TYPE, STABLE_PINS, LATEST_PINS,
    REPO_URLS, INSTALL_SENTINEL, DIR_DEPS,
)
from ..core.env_manager import EnvManager
from ..core.subprocess_runner import SubprocessRunner, pip_install_progress_parser
from ..core.cuda_builder import CudaBuilder

logger = logging.getLogger("4K4D.n02_dependency_install")


class FourK4D_DependencyInstall(BaseEasyVolcapNode):
    """
    Installs all 4K4D / EasyVolcap dependencies.

    This node handles the complete installation process:
    1. Detect Python and CUDA environment
    2. Install PyTorch (if not present)
    3. Install PyTorch3D
    4. Clone EasyVolcap and 4K4D repositories
    5. Install EasyVolcap in editable mode
    6. Install 4K4D in editable mode
    7. Compile CUDA extensions (diff-point-rasterization, tinycudann)
    8. Install auxiliary packages (open3d, gdown, ffmpeg-python, colmap)
    9. Install RobustVideoMatting
    10. Verify all installations

    Uses sentinel files to avoid redundant installations.
    """

    CATEGORY = CATEGORIES["setup"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("dataset_info", "install_log", "all_ready", "python_path")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
                "version_mode": (["stable", "latest"], {"default": "stable"}),
            },
            "optional": {
                "force_reinstall": ("BOOLEAN", {"default": False}),
                "cuda_arch": ("STRING", {"default": "auto"}),
                "skip_cuda_extensions": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def execute(self, dataset_info, version_mode="stable", force_reinstall=False,
                cuda_arch="auto", skip_cuda_extensions=False, unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, version_mode, force_reinstall,
            cuda_arch, skip_cuda_extensions, unique_id
        )

    def _run(self, dataset_info, version_mode, force_reinstall,
             cuda_arch, skip_cuda_extensions, unique_id):
        self._validate_dataset_info(dataset_info)

        runner = self._create_runner()
        env = self.env
        log_lines = []
        all_ready = True

        deps_dir = Path(env.paths["deps"])
        deps_dir.mkdir(parents=True, exist_ok=True)

        pins = STABLE_PINS if version_mode == "stable" else LATEST_PINS
        sentinel = env.node_pack_root / INSTALL_SENTINEL.format(mode=version_mode)

        # Check if already installed
        if sentinel.exists() and not force_reinstall:
            log_lines.append("Dependencies already installed (sentinel found). Skipping.")
            log_lines.append("Set force_reinstall=True to reinstall.")
            return (
                self._update_dataset_info(dataset_info, {
                    "dep_version_mode": version_mode,
                    "easyvolcap_root": str(deps_dir / "EasyVolcap"),
                }),
                "\n".join(log_lines),
                True,
                sys.executable,
            )

        # Step 1: Detect environment
        log_lines.append("=" * 60)
        log_lines.append("STEP 1: Detecting environment...")
        log_lines.append(f"  Python: {sys.executable}")
        log_lines.append(f"  RunPod: {env.is_runpod}")

        gpu_info = env.gpu_info
        log_lines.append(f"  GPU: {gpu_info['gpu_name']}")
        log_lines.append(f"  CUDA: {gpu_info['cuda_version']}")
        log_lines.append(f"  VRAM: {gpu_info['gpu_memory_gb']}GB")

        if cuda_arch == "auto":
            cuda_arch = gpu_info.get("gpu_arch", "8.6")
        log_lines.append(f"  CUDA Arch: sm_{cuda_arch.replace('.', '')}")

        # Step 2: Set CUDA environment
        log_lines.append("\nSTEP 2: Setting CUDA environment variables...")
        cuda_env = env.get_cuda_env_vars()
        log_lines.append(f"  TORCH_CUDA_ARCH_LIST={cuda_env.get('TORCH_CUDA_ARCH_LIST', 'not set')}")

        # Step 3: Check PyTorch
        log_lines.append("\nSTEP 3: Checking PyTorch...")
        try:
            import torch
            log_lines.append(f"  PyTorch {torch.__version__} already installed")
            log_lines.append(f"  CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            log_lines.append("  PyTorch not found — installing...")
            result = runner.run_simple(
                [sys.executable, "-m", "pip", "install",
                 "torch==2.1.0", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu118"],
                timeout=600,
            )
            if not result.success:
                log_lines.append(f"  FAILED: {result.error_summary[:200]}")
                all_ready = False

        # Step 4: Install PyTorch3D
        log_lines.append("\nSTEP 4: Installing PyTorch3D...")
        if not skip_cuda_extensions:
            cuda_builder = CudaBuilder()
            pt3d_result = cuda_builder.compile_pytorch3d()
            log_lines.append(f"  {pt3d_result['message']}")
            if not pt3d_result["success"]:
                all_ready = False

        # Step 5: Clone EasyVolcap
        log_lines.append("\nSTEP 5: Cloning EasyVolcap...")
        easyvolcap_dir = deps_dir / "EasyVolcap"
        if easyvolcap_dir.exists() and not force_reinstall:
            log_lines.append("  EasyVolcap directory exists, checking out correct version...")
            result = runner.run_simple(
                ["git", "checkout", pins["easyvolcap"]],
                cwd=str(easyvolcap_dir),
            )
        else:
            result = runner.run_simple(
                ["git", "clone", REPO_URLS["easyvolcap"], str(easyvolcap_dir)],
                timeout=300,
            )
            if result.success and pins["easyvolcap"] != "main":
                runner.run_simple(
                    ["git", "checkout", pins["easyvolcap"]],
                    cwd=str(easyvolcap_dir),
                )
        log_lines.append(f"  {'OK' if result.success else 'FAILED'}")

        # Step 6: Install EasyVolcap
        log_lines.append("\nSTEP 6: Installing EasyVolcap (editable)...")
        result = runner.run_simple(
            [sys.executable, "-m", "pip", "install", "-e", str(easyvolcap_dir),
             "--no-build-isolation", "--no-deps"],
            env=cuda_env,
            timeout=600,
        )
        log_lines.append(f"  {'OK' if result.success else 'FAILED'}")
        if not result.success:
            all_ready = False

        # Step 7: Clone 4K4D
        log_lines.append("\nSTEP 7: Cloning 4K4D...")
        k4d_dir = deps_dir / "4K4D"
        if k4d_dir.exists() and not force_reinstall:
            log_lines.append("  4K4D directory exists, checking out correct version...")
            result = runner.run_simple(
                ["git", "checkout", pins["4k4d"]],
                cwd=str(k4d_dir),
            )
        else:
            result = runner.run_simple(
                ["git", "clone", REPO_URLS["4k4d"], str(k4d_dir)],
                timeout=300,
            )
        log_lines.append(f"  {'OK' if result.success else 'FAILED'}")

        # Step 8: Install 4K4D
        log_lines.append("\nSTEP 8: Installing 4K4D (editable)...")
        result = runner.run_simple(
            [sys.executable, "-m", "pip", "install", "-e", str(k4d_dir),
             "--no-build-isolation", "--no-deps"],
            env=cuda_env,
            timeout=600,
        )
        log_lines.append(f"  {'OK' if result.success else 'FAILED'}")
        if not result.success:
            all_ready = False

        # Step 9: CUDA extensions
        if not skip_cuda_extensions:
            log_lines.append("\nSTEP 9: Compiling CUDA extensions...")
            cuda_builder = CudaBuilder()

            # diff-point-rasterization
            dpr_result = cuda_builder.compile_diff_point_rasterization(
                REPO_URLS["diff_point_rasterization"],
                pins["diff_point_rasterization"],
            )
            log_lines.append(f"  diff-point-rasterization: {dpr_result['message']}")

            # tinycudann
            tcnn_result = cuda_builder.compile_tinycudann()
            log_lines.append(f"  tinycudann: {tcnn_result['message']}")
        else:
            log_lines.append("\nSTEP 9: CUDA extensions SKIPPED (skip_cuda_extensions=True)")
            log_lines.append("  WARNING: Rendering quality may be reduced without CUDA extensions.")

        # Step 10: Auxiliary packages
        log_lines.append("\nSTEP 10: Installing auxiliary packages...")
        aux_packages = [
            "open3d", "gdown", "ffmpeg-python", "scipy",
            "yapf", "trimesh", "websockets",
            "scikit-image", "tensorboard",
            "pyperclip", "PyTurboJPEG",
            "pyntcloud", "PyMCubes", "imgui-bundle", "PyGLM",
            "opencv-python-headless", "tqdm", "dotdict",
            "ruamel.yaml", "addict", "ujson", "commentjson",
            "yacs", "plyfile", "smplx", "h5py",
            "lpips", "kornia", "einops",
            "pytorch-msssim", "mediapy",
            "cuda-python",  # CUDA-GL interop for headless rendering
        ]
        for pkg in aux_packages:
            result = runner.run_simple(
                [sys.executable, "-m", "pip", "install", pkg],
                timeout=120,
            )
            status = "OK" if result.success else "FAILED"
            log_lines.append(f"  {pkg}: {status}")

        # Step 11: Install COLMAP
        log_lines.append("\nSTEP 11: Checking COLMAP...")
        result = runner.run_simple(["colmap", "--help"], timeout=10)
        if result.success:
            log_lines.append("  COLMAP binary already installed")
        else:
            log_lines.append("  COLMAP binary not found, installing pycolmap...")
            # Skip apt-get entirely — it's unreliable in containers and can
            # stall or interfere with the parent process. Use pycolmap instead.
            colmap_installed = False
            for cmd in [
                [sys.executable, "-m", "pip", "install", "pycolmap"],
                [sys.executable, "-m", "pip", "install", "colmap"],
            ]:
                result = runner.run_simple(cmd, timeout=300)
                if result.success:
                    colmap_installed = True
                    log_lines.append("  Installed pycolmap (Python bindings)")
                    break

            if colmap_installed:
                log_lines.append("  COLMAP: OK")
            else:
                log_lines.append(
                    "  COLMAP: FAILED — Install manually with: "
                    "pip install pycolmap"
                )

        # Step 12: RobustVideoMatting
        log_lines.append("\nSTEP 12: Installing RobustVideoMatting...")
        result = runner.run_simple(
            [sys.executable, "-m", "pip", "install", "robust-video-matting"],
            timeout=120,
        )
        if not result.success:
            # Try from source
            result = runner.run_simple(
                [sys.executable, "-m", "pip", "install",
                 f"git+{REPO_URLS['robust_video_matting']}"],
                timeout=300,
            )
        log_lines.append(f"  {'OK' if result.success else 'FAILED'}")

        # Step 12b: Patch pyntcloud for pandas >= 2.0 compatibility
        # pyntcloud 0.3.x uses df.dtypes[i] which does label-based lookup
        # in pandas 2.0+, causing KeyError. Fix: use df.dtypes.iloc[i]
        log_lines.append("\nSTEP 12b: Patching pyntcloud for pandas compatibility...")
        patch_code = (
            "import pyntcloud.io.ply as m, inspect, pathlib; "
            "src = inspect.getfile(m); "
            "txt = pathlib.Path(src).read_text(); "
            "old = 'property_formats[str(df.dtypes[i])[0]]'; "
            "new = 'property_formats[str(df.dtypes.iloc[i])[0]]'; "
            "("
            "  pathlib.Path(src).write_text(txt.replace(old, new)), "
            "  print('Patched' if old in txt else 'Already patched')"
            ")"
        )
        result = runner.run_simple(
            [sys.executable, "-c", patch_code],
            timeout=30,
        )
        log_lines.append(f"  pyntcloud patch: {'OK' if result.success else 'FAILED'}")

        # Step 13: Verify installations
        log_lines.append("\nSTEP 13: Verifying installations...")
        verifications = {
            "torch": "import torch; print(f'PyTorch {torch.__version__}')",
            "easyvolcap": "from easyvolcap.utils.console_utils import *; print('EasyVolcap OK')",
            "numpy": "import numpy; print(f'NumPy {numpy.__version__}')",
        }
        for name, check_code in verifications.items():
            result = runner.run_simple(
                [sys.executable, "-c", check_code],
                timeout=30,
            )
            status = "OK" if result.success else "FAILED"
            log_lines.append(f"  {name}: {status}")
            if not result.success and name in ("torch",):
                all_ready = False

        # Write sentinel
        if all_ready:
            sentinel.write_text(f"installed_{version_mode}")
            log_lines.append(f"\nInstallation complete! Sentinel written to {sentinel}")
        else:
            log_lines.append("\nWARNING: Some installations failed. Check log above.")

        log_lines.append("=" * 60)

        updated_info = self._update_dataset_info(dataset_info, {
            "dep_version_mode": version_mode,
            "easyvolcap_root": str(easyvolcap_dir),
            "env_info": env.get_env_info_dict(),
        })

        return (
            updated_info,
            "\n".join(log_lines),
            all_ready,
            sys.executable,
        )

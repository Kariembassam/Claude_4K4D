"""
ComfyUI-4K4D Constants
=====================
Central constants, type definitions, version pins, and configuration defaults
used across all nodes and core modules.
"""

__version__ = "2.0.0"

# ─── ComfyUI Custom Type Names ───────────────────────────────────────────────
DATASET_INFO_TYPE = "DATASET_INFO"

# ─── Node Categories ─────────────────────────────────────────────────────────
CATEGORY_PREFIX = "4K4D"
CATEGORIES = {
    "input": f"{CATEGORY_PREFIX}/Input",
    "setup": f"{CATEGORY_PREFIX}/Setup",
    "preprocessing": f"{CATEGORY_PREFIX}/Preprocessing",
    "processing": f"{CATEGORY_PREFIX}/Processing",
    "training": f"{CATEGORY_PREFIX}/Training",
    "rendering": f"{CATEGORY_PREFIX}/Rendering",
    "output": f"{CATEGORY_PREFIX}/Output",
    "utilities": f"{CATEGORY_PREFIX}/Utilities",
}

# ─── Version Pins ─────────────────────────────────────────────────────────────
# Stable: pinned commit hashes (tested, known-good)
STABLE_PINS = {
    "easyvolcap": "main",  # Pin to specific commit at release
    "4k4d": "main",        # Pin to specific commit at release
    "diff_point_rasterization": "main",
}

# Latest: main branch HEAD (may break, cutting edge)
LATEST_PINS = {
    "easyvolcap": "main",
    "4k4d": "main",
    "diff_point_rasterization": "main",
}

# ─── Source Repository URLs ───────────────────────────────────────────────────
REPO_URLS = {
    "easyvolcap": "https://github.com/zju3dv/EasyVolcap.git",
    "4k4d": "https://github.com/zju3dv/4K4D.git",
    "diff_point_rasterization": "https://github.com/dendenxu/diff-point-rasterization.git",
    "robust_video_matting": "https://github.com/PeterL1n/RobustVideoMatting.git",
}

# ─── Supported Formats ───────────────────────────────────────────────────────
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".exr", ".bmp"}
SUPPORTED_VIDEO_CODECS = {"h264", "hevc", "h265", "prores", "dnxhd", "mjpeg", "vp9", "av1"}

# ─── Directory Names ──────────────────────────────────────────────────────────
DIR_IMAGES = "images"
DIR_MASKS = "masks"
DIR_VHULLS = "vhulls"
DIR_SURFS = "surfs"
DIR_LOGS = "logs"
DIR_SENTINELS = ".sentinels"
DIR_COMPILED = ".compiled"
DIR_DEPS = "deps"
DIR_DATA = "data"
DIR_EXPORTS = "exports"
DIR_CONFIGS = "configs"

# ─── File Naming ──────────────────────────────────────────────────────────────
FRAME_NAME_FORMAT = "{:06d}.jpg"  # 000000.jpg, 000001.jpg, ...
SENTINEL_EXTENSION = ".json"
INSTALL_SENTINEL = ".install_complete_{mode}"

# ─── Default Parameters ──────────────────────────────────────────────────────
DEFAULTS = {
    "dataset_name": "my_sequence",
    "camera_count": 5,
    "resolution_scale": 0.5,
    "fps": 0.0,  # 0.0 = auto-detect
    "vhull_thresh_sparse": 0.75,   # <8 cameras
    "vhull_thresh_dense": 0.90,    # 8+ cameras
    "bbox_padding": 0.3,           # metres
    "focal_ratio": 0.5,            # wider for multi-person
    "preview_iterations": 200,
    "full_iterations": 1600,
    "checkpoint_interval": 100,
    "sync_max_offset_frames": 10,
    "min_mask_confidence": 0.75,
    "max_blur_fraction": 0.10,
    "min_sync_quality": 0.85,
    "crf": 18,  # H.264 quality (lower = better)
    "bg_brightness": 0.0,
}

# ─── PSNR Thresholds ─────────────────────────────────────────────────────────
PSNR_WARN_THRESHOLD = 20.0   # Warn if PSNR < 20 dB at iter 100
PSNR_ERROR_THRESHOLD = 15.0  # Error if PSNR < 15 dB at iter 100
PSNR_EXPECTED_PREVIEW = 24.0 # Expected PSNR at iter 200 for preview

# ─── CUDA Configuration ──────────────────────────────────────────────────────
RTX_4090_CUDA_ARCH = "8.6"
SUPPORTED_CUDA_VERSIONS = ["11.8", "12.1", "12.4"]

# ─── Dataset Info Schema Template ─────────────────────────────────────────────
def create_empty_dataset_info() -> dict:
    """Create a new empty dataset_info dict with all fields initialized."""
    return {
        "dataset_name": "",
        "dataset_root": "",
        "video_paths": [],
        "camera_count": 0,
        "detected_fps": 0.0,
        "detected_resolution": "",
        "detected_format": "",
        "has_calibration": False,
        "has_masks": False,
        "sequence_length": None,
        "bounds": None,
        "model_path": None,
        "supercharged_path": None,
        "render_output": None,
        "quality_gate_passed": False,
        "dep_version_mode": "stable",
        "background_mode": "foreground_only",
        "config_path": None,
        "experiment_name": None,
        "easyvolcap_root": None,
        "env_info": {},
        "logs_dir": "",
        "sentinel_dir": "",
        "errors": [],
        "warnings": [],
    }

"""
ComfyUI-4K4D Dataset Structure
================================
Helpers for creating and validating the EasyVolcap-compatible
directory layout that 4K4D expects.
"""

import logging
from pathlib import Path

from .constants import DIR_IMAGES, DIR_MASKS, DIR_VHULLS, DIR_SURFS, DIR_SENTINELS

logger = logging.getLogger("4K4D.dataset_structure")


def create_dataset_dirs(dataset_root: str, camera_count: int) -> dict:
    """
    Create the full EasyVolcap directory structure for a dataset.

    Creates:
        data/{name}/
        ├── images/
        │   ├── 00/
        │   ├── 01/
        │   └── ...
        ├── masks/
        │   ├── 00/
        │   ├── 01/
        │   └── ...
        ├── vhulls/
        ├── surfs/
        └── .sentinels/

    Returns:
        dict mapping directory names to their absolute paths
    """
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    dirs = {
        "root": str(root),
        "images": str(root / DIR_IMAGES),
        "masks": str(root / DIR_MASKS),
        "vhulls": str(root / DIR_VHULLS),
        "surfs": str(root / DIR_SURFS),
        "sentinels": str(root / DIR_SENTINELS),
        "camera_dirs": [],
        "mask_dirs": [],
    }

    # Create main directories
    for d in [DIR_IMAGES, DIR_MASKS, DIR_VHULLS, DIR_SURFS, DIR_SENTINELS]:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Create per-camera subdirectories
    for cam_idx in range(camera_count):
        cam_name = f"{cam_idx:02d}"

        img_dir = root / DIR_IMAGES / cam_name
        img_dir.mkdir(parents=True, exist_ok=True)
        dirs["camera_dirs"].append(str(img_dir))

        mask_dir = root / DIR_MASKS / cam_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        dirs["mask_dirs"].append(str(mask_dir))

    logger.info(f"Created dataset structure at {root} with {camera_count} cameras")
    return dirs


def validate_dataset_structure(dataset_root: str) -> dict:
    """
    Validate an existing dataset directory structure.

    Returns:
        dict with keys:
        - valid (bool): whether the structure is valid
        - camera_count (int): number of camera directories found
        - frame_counts (dict): {cam_id: frame_count}
        - issues (list): list of issues found
        - warnings (list): list of warnings
    """
    root = Path(dataset_root)
    result = {
        "valid": True,
        "camera_count": 0,
        "frame_counts": {},
        "has_images": False,
        "has_masks": False,
        "has_calibration": False,
        "has_vhulls": False,
        "issues": [],
        "warnings": [],
    }

    if not root.exists():
        result["valid"] = False
        result["issues"].append(f"Dataset root does not exist: {dataset_root}")
        return result

    # Check images
    images_dir = root / DIR_IMAGES
    if images_dir.is_dir():
        cam_dirs = sorted([
            d for d in images_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        result["camera_count"] = len(cam_dirs)
        result["has_images"] = len(cam_dirs) > 0

        for cam_dir in cam_dirs:
            frames = sorted([
                f for f in cam_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            result["frame_counts"][cam_dir.name] = len(frames)

        # Check frame count consistency
        counts = list(result["frame_counts"].values())
        if counts and len(set(counts)) > 1:
            result["warnings"].append(
                f"Inconsistent frame counts across cameras: {result['frame_counts']}. "
                "All cameras should have the same number of frames."
            )
    else:
        result["issues"].append("No images/ directory found")

    # Check masks
    masks_dir = root / DIR_MASKS
    if masks_dir.is_dir():
        mask_cam_dirs = [d for d in masks_dir.iterdir() if d.is_dir()]
        result["has_masks"] = len(mask_cam_dirs) > 0
        if result["has_masks"] and len(mask_cam_dirs) != result["camera_count"]:
            result["warnings"].append(
                f"Mask directories ({len(mask_cam_dirs)}) don't match "
                f"camera count ({result['camera_count']})"
            )

    # Check calibration
    result["has_calibration"] = (
        (root / "extri.yml").exists() and (root / "intri.yml").exists()
    )

    # Check visual hulls
    result["has_vhulls"] = (root / DIR_VHULLS).is_dir() and any(
        (root / DIR_VHULLS).iterdir()
    ) if (root / DIR_VHULLS).is_dir() else False

    # Camera count warnings
    if result["camera_count"] > 0 and result["camera_count"] < 8:
        result["warnings"].append(
            f"Only {result['camera_count']} cameras detected. 4K4D works best "
            "with 8+ cameras. Results with fewer cameras may have artifacts at "
            "large viewing angle changes."
        )

    if result["issues"]:
        result["valid"] = False

    return result

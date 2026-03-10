"""
ComfyUI-4K4D Node 6: Visual Hull
==================================
Space carving from calibrated cameras + masks to generate
initial 3D point cloud and auto bounding box.
"""

import json
import logging
import os
from pathlib import Path

from core.base_node import BaseEasyVolcapNode
from core.constants import CATEGORIES, DATASET_INFO_TYPE, DEFAULTS
from core.checkpoint_manager import CheckpointManager
from core.config_generator import ConfigGenerator

logger = logging.getLogger("4K4D.n06_visual_hull")


class FourK4D_VisualHull(BaseEasyVolcapNode):
    """
    Computes visual hull from calibrated cameras and masks.

    Produces:
    - vhulls/ directory with visual hull volumes
    - surfs/ directory with surface meshes
    - Auto-generated {name}_obj.yaml with tight bounding box
    """

    CATEGORY = CATEGORIES["processing"]
    FUNCTION = "execute"
    RETURN_TYPES = (DATASET_INFO_TYPE, "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("dataset_info", "hull_preview", "bounding_box_str", "point_count")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "vhull_thresh": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05}),
                "bbox_padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
                "auto_set_thresh_by_camera_count": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, vhull_thresh=0.75, bbox_padding=0.3,
                auto_set_thresh_by_camera_count=True, unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, vhull_thresh, bbox_padding,
            auto_set_thresh_by_camera_count, unique_id
        )

    def _run(self, dataset_info, vhull_thresh, bbox_padding,
             auto_set_thresh_by_camera_count, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root", "dataset_name"])

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info["dataset_name"]
        camera_count = dataset_info.get("camera_count", 5)

        cm = CheckpointManager(os.path.join(dataset_root, ".sentinels"))
        if cm.is_completed("visual_hull"):
            meta = cm.get_metadata("visual_hull")
            return (
                self._update_dataset_info(dataset_info, {
                    "bounds": meta.get("bounds"),
                }),
                self._create_error_image("Cached"),
                json.dumps(meta.get("bounds", [])),
                meta.get("point_count", 0),
            )

        cm.mark_started("visual_hull")
        runner = self._create_runner()

        # Auto-adjust threshold based on camera count
        if auto_set_thresh_by_camera_count:
            if camera_count < 8:
                vhull_thresh = DEFAULTS["vhull_thresh_sparse"]
            else:
                vhull_thresh = DEFAULTS["vhull_thresh_dense"]

        self._node_logger.info(f"Running visual hull with thresh={vhull_thresh}, padding={bbox_padding}m")

        # Generate dataset config if not exists
        config_path = dataset_info.get("config_path")
        if not config_path or not os.path.exists(str(config_path)):
            try:
                gen = ConfigGenerator()
                config_path = gen.generate_dataset_config(dataset_info)
            except Exception as e:
                self._node_logger.warning(f"Config generation failed: {e}")
                config_path = None

        # Run visual hull extraction
        easyvolcap_root = dataset_info.get("easyvolcap_root", "")
        point_count = 0
        bounds = [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]  # Default bounds

        if easyvolcap_root and os.path.exists(easyvolcap_root):
            # Use evc-test for vhulls
            vhull_configs = []
            if config_path:
                vhull_configs.append(str(config_path))
            vhull_configs.extend([
                "configs/specs/optimized.yaml",
                "configs/specs/vhulls.yaml",
            ])

            result = runner.run(
                self.build_evc_command("evc-test", vhull_configs),
                cwd=easyvolcap_root,
                unique_id=unique_id,
                timeout_seconds=3600,
            )

            if result.success:
                self._node_logger.info("Visual hull extraction completed")

                # Run surface extraction
                surf_configs = list(vhull_configs)
                surf_configs[-1] = "configs/specs/surfs.yaml"
                runner.run(
                    self.build_evc_command("evc-test", surf_configs),
                    cwd=easyvolcap_root,
                    timeout_seconds=3600,
                )
        else:
            self._node_logger.warning("EasyVolcap not found — generating estimated bounds from calibration")

        # Try to extract bounds from generated visual hull
        vhulls_dir = os.path.join(dataset_root, "vhulls")
        if os.path.exists(vhulls_dir):
            try:
                bounds, point_count = self._extract_bounds(vhulls_dir, bbox_padding)
            except Exception as e:
                self._node_logger.warning(f"Bounds extraction failed: {e}")

        # Generate {name}_obj.yaml with bounds
        try:
            gen = ConfigGenerator()
            obj_config = gen.generate_dataset_obj_config(
                self._update_dataset_info(dataset_info, {"bounds": bounds})
            )
            self._node_logger.info(f"Generated object config: {obj_config}")
        except Exception as e:
            self._node_logger.warning(f"Object config generation failed: {e}")

        bounds_str = json.dumps(bounds)

        cm.mark_completed("visual_hull", {
            "bounds": bounds,
            "point_count": point_count,
            "vhull_thresh": vhull_thresh,
        })

        preview = self._create_error_image(f"Visual Hull: {point_count} points")

        return (
            self._update_dataset_info(dataset_info, {
                "bounds": bounds,
                "config_path": config_path,
            }),
            preview,
            bounds_str,
            point_count,
        )

    def _extract_bounds(self, vhulls_dir, padding):
        """Extract bounding box from visual hull PLY files."""
        try:
            import numpy as np

            # Look for PLY or NPZ files
            points = []
            for f in Path(vhulls_dir).rglob("*.ply"):
                # Simple PLY reader for vertex positions
                pts = self._read_ply_vertices(str(f))
                if pts is not None:
                    points.append(pts)

            if not points:
                return [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], 0

            all_points = np.concatenate(points, axis=0)
            point_count = len(all_points)

            min_bounds = all_points.min(axis=0) - padding
            max_bounds = all_points.max(axis=0) + padding

            bounds = [min_bounds.tolist(), max_bounds.tolist()]
            return bounds, point_count

        except Exception:
            return [[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], 0

    def _read_ply_vertices(self, ply_path):
        """Read vertex positions from a PLY file."""
        try:
            import numpy as np

            with open(ply_path, 'r') as f:
                header = True
                vertex_count = 0
                vertices = []

                for line in f:
                    line = line.strip()
                    if header:
                        if line.startswith("element vertex"):
                            vertex_count = int(line.split()[-1])
                        elif line == "end_header":
                            header = False
                    else:
                        parts = line.split()
                        if len(parts) >= 3:
                            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            if len(vertices) >= vertex_count:
                                break

            if vertices:
                return np.array(vertices)
        except Exception:
            pass
        return None

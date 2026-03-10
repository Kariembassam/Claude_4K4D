"""
ComfyUI-4K4D: Real-Time 4D View Synthesis Node Pack
=====================================================
Production-grade ComfyUI custom node pack wrapping the full
4K4D / EasyVolcap pipeline (CVPR 2024) into automated,
visual, node-based workflows.

14 nodes covering the complete pipeline:
  Input & Setup:    FolderIngest, DependencyInstall, VersionManager
  Preprocessing:    FrameExtract, CameraCalibration, MaskGeneration,
                    VisualHull, QualityGate
  Training:         Train, SuperCharge, Render
  Output:           Viewer, StatusMonitor, ExportPack, Cleanup

Target: RunPod RTX 4090 (24GB VRAM) via official ComfyUI template.
"""

from .node_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

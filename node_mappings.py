"""
ComfyUI-4K4D Node Mappings
============================
Central registry mapping node class names to their implementations.
ComfyUI reads NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS from this module.
"""

from nodes.n01_folder_ingest import FourK4D_FolderIngest
from nodes.n02_dependency_install import FourK4D_DependencyInstall
from nodes.n03_frame_extract import FourK4D_FrameExtract
from nodes.n04_camera_calibration import FourK4D_CameraCalibration
from nodes.n05_mask_generation import FourK4D_MaskGeneration
from nodes.n06_visual_hull import FourK4D_VisualHull
from nodes.n06b_quality_gate import FourK4D_QualityGate
from nodes.n07_train import FourK4D_Train
from nodes.n08_supercharge import FourK4D_SuperCharge
from nodes.n09_render import FourK4D_Render
from nodes.n10_viewer import FourK4D_Viewer
from nodes.n11_status_monitor import FourK4D_StatusMonitor
from nodes.n12_export_pack import FourK4D_ExportPack
from nodes.n13_version_manager import FourK4D_VersionManager
from nodes.n14_cleanup import FourK4D_Cleanup

NODE_CLASS_MAPPINGS = {
    "FourK4D_FolderIngest": FourK4D_FolderIngest,
    "FourK4D_DependencyInstall": FourK4D_DependencyInstall,
    "FourK4D_FrameExtract": FourK4D_FrameExtract,
    "FourK4D_CameraCalibration": FourK4D_CameraCalibration,
    "FourK4D_MaskGeneration": FourK4D_MaskGeneration,
    "FourK4D_VisualHull": FourK4D_VisualHull,
    "FourK4D_QualityGate": FourK4D_QualityGate,
    "FourK4D_Train": FourK4D_Train,
    "FourK4D_SuperCharge": FourK4D_SuperCharge,
    "FourK4D_Render": FourK4D_Render,
    "FourK4D_Viewer": FourK4D_Viewer,
    "FourK4D_StatusMonitor": FourK4D_StatusMonitor,
    "FourK4D_ExportPack": FourK4D_ExportPack,
    "FourK4D_VersionManager": FourK4D_VersionManager,
    "FourK4D_Cleanup": FourK4D_Cleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FourK4D_FolderIngest": "4K4D Folder Ingest",
    "FourK4D_DependencyInstall": "4K4D Dependency Install",
    "FourK4D_FrameExtract": "4K4D Frame Extract",
    "FourK4D_CameraCalibration": "4K4D Camera Calibration",
    "FourK4D_MaskGeneration": "4K4D Mask Generation",
    "FourK4D_VisualHull": "4K4D Visual Hull",
    "FourK4D_QualityGate": "4K4D Quality Gate",
    "FourK4D_Train": "4K4D Train",
    "FourK4D_SuperCharge": "4K4D SuperCharge",
    "FourK4D_Render": "4K4D Render",
    "FourK4D_Viewer": "4K4D Viewer",
    "FourK4D_StatusMonitor": "4K4D Status Monitor",
    "FourK4D_ExportPack": "4K4D Export Pack",
    "FourK4D_VersionManager": "4K4D Version Manager",
    "FourK4D_Cleanup": "4K4D Cleanup",
}

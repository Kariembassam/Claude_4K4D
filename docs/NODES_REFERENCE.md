# ComfyUI-4K4D Nodes Reference

Complete reference for all 14 nodes in the ComfyUI-4K4D pipeline.

---

## Node 1: FolderIngest

**Category**: `4K4D/Input`
**Class**: `FourK4D_FolderIngest`

Scans an input directory and initializes the DATASET_INFO pipeline data.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| input_path | STRING | Yes | — | Path to folder containing multi-camera video files |
| dataset_name | STRING | Yes | — | Name for this dataset |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Initialized pipeline data dictionary |

---

## Node 2: DependencyInstall

**Category**: `4K4D/Setup`
**Class**: `FourK4D_DependencyInstall`

Installs all required dependencies (PyTorch3D, EasyVolcap, 4K4D, COLMAP, CUDA extensions).

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data from previous node |
| version_mode | COMBO | Yes | `stable` | `stable` or `latest` |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with dependency paths |

---

## Node 3: FrameExtract

**Category**: `4K4D/Preprocessing`
**Class**: `FourK4D_FrameExtract`

Extracts frames from video files, with optional transcoding and sync alignment.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| target_fps | INT | No | 30 | Output framerate |
| resolution_scale | FLOAT | No | 1.0 | Scale factor (0.25–2.0) |
| sync_method | COMBO | No | `auto` | `auto`, `audio_xcorr`, `none` |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with frame paths and counts |

---

## Node 4: CameraCalibration

**Category**: `4K4D/Preprocessing`
**Class**: `FourK4D_CameraCalibration`

Runs COLMAP or loads existing camera calibration files.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| method | COMBO | Yes | `auto` | `auto` or `existing` |
| matching_type | COMBO | No | `exhaustive` | `exhaustive` or `sequential` |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with calibration status |

---

## Node 5: MaskGeneration

**Category**: `4K4D/Preprocessing`
**Class**: `FourK4D_MaskGeneration`

Generates foreground masks using RobustVideoMatting or loads existing masks.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| method | COMBO | Yes | `auto` | `auto`, `existing`, `skip` |
| dilation | INT | No | 5 | Mask edge dilation (pixels) |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with mask status |

---

## Node 6: VisualHull

**Category**: `4K4D/Processing`
**Class**: `FourK4D_VisualHull`

Performs space carving to determine 3D bounding box of the subject.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| resolution | INT | No | 256 | Voxel grid resolution |
| vhull_thresh | FLOAT | No | 0.5 | Occupancy threshold |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with bounds and visual hull |

---

## Node 6b: QualityGate

**Category**: `4K4D/Preprocessing`
**Class**: `FourK4D_QualityGate`

**MANDATORY** checkpoint before training. Runs 4 quality checks and blocks on failure.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| force_pass | BOOLEAN | No | False | Override quality gate |
| confirmation | STRING | No | "" | Must be "I UNDERSTAND" to force-pass |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with quality gate status |

### Quality Checks
1. **Mask Quality** — Validates bimodal (black/white) distribution
2. **Blur/Sharpness** — Measures Laplacian variance per frame
3. **Sync Alignment** — Verifies cross-correlation confidence
4. **Camera Coverage** — Assesses spatial distribution (info only)

---

## Node 7: Train

**Category**: `4K4D/Training`
**Class**: `FourK4D_Train`

Trains the 4K4D model. Most complex node with checkpoint/resume support.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| mode | COMBO | Yes | `preview_static` | `preview_static` or `full_sequence` |
| max_iterations | INT | No | 1600 | Training iterations |
| learning_rate | FLOAT | No | 0.0007 | Learning rate |
| checkpoint_interval | INT | No | 100 | Save every N iterations |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with training metrics |

---

## Node 8: SuperCharge

**Category**: `4K4D/Processing`
**Class**: `FourK4D_SuperCharge`

Converts trained model to real-time rendering format using charger.py.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| include_background | BOOLEAN | No | True | Include background model |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with supercharged model path |

---

## Node 9: Render

**Category**: `4K4D/Rendering`
**Class**: `FourK4D_Render`

Renders novel view videos from the trained model.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| mode | COMBO | Yes | `spiral` | `spiral`, `test_views`, `free_viewpoint` |
| output_format | COMBO | No | `h264` | `h264`, `prores`, `png_sequence` |
| fps | INT | No | 30 | Output framerate |
| num_frames | INT | No | 120 | Number of frames to render |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Updated with render output path |

---

## Node 10: Viewer

**Category**: `4K4D/Output`
**Class**: `FourK4D_Viewer`

4-tab viewer: Video, WebGL 3D, Split comparison, iFrame.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| default_tab | COMBO | No | `video` | Initial active tab |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| DATASET_INFO | DATASET_INFO | Pass-through |

---

## Node 11: StatusMonitor

**Category**: `4K4D/Utilities`
**Class**: `FourK4D_StatusMonitor`

Displays live training metrics, log tail, disk usage, and ETA.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| tail_lines | INT | No | 50 | Log lines to display |
| show_gpu | BOOLEAN | No | True | Show GPU metrics |

---

## Node 12: ExportPack

**Category**: `4K4D/Output`
**Class**: `FourK4D_ExportPack`

Packages outputs for NLE import (Premiere Pro, DaVinci Resolve).

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| target_nle | COMBO | Yes | `premiere` | `premiere` or `resolve` |
| include_proxies | BOOLEAN | No | True | Generate proxy media |

---

## Node 13: VersionManager

**Category**: `4K4D/Setup`
**Class**: `FourK4D_VersionManager`

Manages stable vs latest dependency versions.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| version_mode | COMBO | Yes | `stable` | `stable` or `latest` |
| force_reinstall | BOOLEAN | No | False | Force full reinstall |

---

## Node 14: Cleanup

**Category**: `4K4D/Utilities`
**Class**: `FourK4D_Cleanup`

Removes intermediate files with dry-run safety default.

### Inputs
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| dataset_info | DATASET_INFO | Yes | — | Pipeline data |
| dry_run | BOOLEAN | Yes | True | Preview without deleting |
| archive_first | BOOLEAN | No | True | Archive before deletion |
| targets | COMBO | No | `intermediates` | What to clean |

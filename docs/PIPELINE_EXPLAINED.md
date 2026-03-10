# ComfyUI-4K4D Pipeline Explained

## Overview

The ComfyUI-4K4D pipeline transforms multi-camera video footage into real-time renderable 4D (3D + time) neural representations using the 4K4D method (CVPR 2024). The pipeline is implemented as 14 ComfyUI nodes that connect in sequence.

## Pipeline Flow

```
Input Videos → Frame Extract → Camera Calibration → Mask Generation
    → Visual Hull → Quality Gate → Training → SuperCharge → Render → Export
```

## Stage-by-Stage Breakdown

### Stage 1: Data Ingestion (Node 1)

**FolderIngest** scans your input directory and detects:
- Video files (MP4, MOV, AVI, MKV)
- Existing calibration files (extri.yml, intri.yml)
- Pre-existing masks
- Dataset structure

It builds the initial `DATASET_INFO` dictionary that flows through every subsequent node.

### Stage 2: Environment Setup (Node 2)

**DependencyInstall** ensures all required software is available:
- PyTorch with CUDA support
- PyTorch3D (compiled from source for your GPU)
- EasyVolcap framework
- 4K4D model code
- CUDA extensions (diff-point-rasterization, tinycudann)
- COLMAP for camera calibration
- RobustVideoMatting for mask generation

Uses sentinel files to skip already-completed steps.

### Stage 3: Frame Extraction (Node 3)

**FrameExtract** processes input videos:
1. Detects format, codec, resolution, and FPS via ffprobe
2. Transcodes to H.264 if needed (ProRes, HEVC sources)
3. Performs audio cross-correlation sync if cameras are not synchronized
4. Extracts frames to `images/00/`, `images/01/`, etc.
5. Uses zero-padded naming: `000000.jpg`, `000001.jpg`, etc.

### Stage 4: Camera Calibration (Node 4)

**CameraCalibration** establishes camera positions:
- **Auto mode**: Runs COLMAP feature extraction + matching + reconstruction
  - Uses exhaustive matching for fixed camera rigs
  - Uses sequential matching for handheld setups
- **Existing mode**: Accepts pre-computed `extri.yml` + `intri.yml`
- Generates EasyVolcap-format camera parameter files

### Stage 5: Mask Generation (Node 5)

**MaskGeneration** creates foreground/background masks:
- **Auto mode**: Uses RobustVideoMatting (RVM) in recurrent mode
  - Handles multiple people in scene
  - Applies configurable dilation for edge cleanup
- **Existing mode**: Uses user-supplied mask directories
- **Skip mode**: No masking (for full-scene reconstruction)

Sets `background_mode` to either `foreground_only` or `full_scene`.

### Stage 6: Visual Hull (Node 6)

**VisualHull** performs space carving:
1. Projects masks from all cameras to compute 3D occupancy
2. Generates tight bounding box around the subject
3. Creates the `{name}_obj.yaml` config with bounds
4. Auto-adjusts `vhull_thresh` based on camera count
   - Sparse (<8 cameras): lower threshold for more generous bounds
   - Dense (8+ cameras): higher threshold for tighter bounds

### Stage 7: Quality Gate (Node 6b)

**QualityGate** is a MANDATORY checkpoint before training. It runs 4 checks:

1. **Mask Quality**: Validates bimodal distribution (mostly black/white)
2. **Blur/Sharpness**: Measures Laplacian variance, flags blurry frames
3. **Sync Alignment**: Verifies cross-correlation confidence
4. **Camera Coverage**: Assesses spatial coverage (info only, never blocks)

If any check fails, training is BLOCKED. Override requires `force_pass=True` and typing "I UNDERSTAND".

### Stage 8: Training (Node 7)

**Train** is the most complex node. Two modes:

#### Preview Static (Quick Test)
- 200 iterations, ~10 minutes
- Single-frame reconstruction
- Validates the pipeline works before committing to full training

#### Full Sequence (Production)
- 1600+ iterations, ~24 hours on RTX 4090
- Full temporal sequence training
- Checkpoint every N iterations (configurable)
- Auto-resume from latest checkpoint
- PSNR monitoring with early warnings:
  - < 20dB at iteration 100 → Warning
  - < 15dB at iteration 100 → Auto-halt

### Stage 9: SuperCharge (Node 8)

**SuperCharge** converts the trained model to the real-time format:
- Runs `charger.py` from 4K4D codebase
- Produces `SuperChargedR4DV` (foreground only) or `SuperChargedR4DVB` (with background)
- Enables real-time rendering at 4K resolution

### Stage 10: Rendering (Node 9)

**Render** generates novel view videos:

| Mode | Description |
|------|-------------|
| `spiral` | 360-degree orbit around subject |
| `test_views` | Render from held-out camera positions |
| `free_viewpoint` | Custom camera trajectory |

Output formats: H.264 (web), ProRes 4444 (NLE), PNG sequence.

### Stage 11: Output (Nodes 10-12)

- **Viewer** (Node 10): 4-tab viewer (Video, WebGL 3D, Split comparison, iFrame)
- **StatusMonitor** (Node 11): Live training metrics, disk usage, ETA
- **ExportPack** (Node 12): Packages outputs for Premiere Pro/DaVinci Resolve

### Utilities (Nodes 13-14)

- **VersionManager** (Node 13): Toggle between stable/latest dependency versions
- **Cleanup** (Node 14): Remove intermediate files, with dry-run safety

## The DATASET_INFO Flow

The `DATASET_INFO` is a typed dictionary that flows through the entire pipeline. Each node reads required keys, performs its operation, and returns an updated copy.

Key fields accumulated through the pipeline:
- `dataset_name`, `dataset_root` (Node 1)
- `camera_count`, `sequence_length` (Node 1/3)
- `has_calibration` (Node 4)
- `has_masks`, `background_mode` (Node 5)
- `has_visual_hull`, `bounds` (Node 6)
- `quality_gate_passed` (Node 6b)
- `training_complete`, `psnr` (Node 7)
- `supercharged` (Node 8)
- `render_output_path` (Node 9)

## Error Handling

Every node is wrapped in `_safe_execute()` which:
1. Catches ALL exceptions (never crashes ComfyUI)
2. Logs detailed errors to `logs/{node}_{timestamp}.log`
3. Returns user-friendly error messages
4. Preserves the DATASET_INFO for debugging

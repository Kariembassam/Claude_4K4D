# ComfyUI-4K4D

**Production-grade ComfyUI custom node pack for real-time 4D view synthesis using the 4K4D method (CVPR 2024).**

Transform multi-camera video footage into real-time renderable 4D neural representations — entirely within ComfyUI's visual workflow editor. No command-line expertise required.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue)](https://github.com/comfyanonymous/ComfyUI)
[![4K4D](https://img.shields.io/badge/4K4D-CVPR%202024-green)](https://zju3dv.github.io/4k4d/)

## Overview

ComfyUI-4K4D wraps the complete [4K4D/EasyVolcap](https://github.com/zju3dv/4K4D) pipeline into **14 automated nodes** that handle everything from raw multi-camera video to renderable 4D assets:

```
📁 Input Videos → 🎬 Frame Extract → 📐 Camera Calibration → 🎭 Mask Generation
→ 📦 Visual Hull → ✅ Quality Gate → 🏋️ Training → ⚡ SuperCharge
→ 🎥 Render → 👁️ Viewer → 📤 Export
```

## Features

- **14 Nodes** covering the complete 4K4D pipeline
- **One-click dependency install** — Node 2 handles PyTorch3D, COLMAP, CUDA extensions
- **Checkpoint/resume** for long-running operations (training survives pod restarts)
- **Quality Gate** blocks training on bad data (saves hours of wasted GPU time)
- **RunPod RTX 4090 optimized** — tested on 24GB VRAM
- **Never crashes ComfyUI** — all exceptions caught and reported gracefully
- **Tiered UI** — simple params visible by default, advanced options in OPTIONAL
- **NLE Export** — packages output for Premiere Pro and DaVinci Resolve

## Quick Start

### Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Kariembassam/Claude_4K4D.git ComfyUI-4K4D
cd ComfyUI-4K4D
pip install -r requirements.txt
```

Restart ComfyUI. The 14 nodes appear under the **4K4D/** category.

### First Run

1. Load `workflows/01_quick_test.json` from the workflow menu
2. Set the input path to your multi-camera footage folder
3. Queue the workflow — the DependencyInstall node handles everything else

### Full Pipeline

Load `workflows/05_full_pipeline.json` for the complete 14-node pipeline with RunPod RTX 4090 defaults.

## Node List

| # | Node | Category | Purpose |
|---|------|----------|---------|
| 1 | FolderIngest | Input | Scan input directory, initialize pipeline |
| 2 | DependencyInstall | Setup | Install PyTorch3D, EasyVolcap, COLMAP, etc. |
| 3 | FrameExtract | Preprocessing | Extract frames, transcode, sync cameras |
| 4 | CameraCalibration | Preprocessing | COLMAP auto or existing calibration |
| 5 | MaskGeneration | Preprocessing | RobustVideoMatting or existing masks |
| 6 | VisualHull | Processing | Space carving, bounding box |
| 6b | QualityGate | Preprocessing | **Mandatory** quality checks before training |
| 7 | Train | Training | 4K4D model training with checkpoint/resume |
| 8 | SuperCharge | Processing | Convert to real-time rendering format |
| 9 | Render | Rendering | Novel view synthesis (spiral, test, free) |
| 10 | Viewer | Output | 4-tab viewer (video, WebGL, split, iframe) |
| 11 | StatusMonitor | Utilities | Live metrics, disk usage, ETA |
| 12 | ExportPack | Output | Package for Premiere Pro / DaVinci Resolve |
| 13 | VersionManager | Setup | Toggle stable/latest dependencies |
| 14 | Cleanup | Utilities | Remove intermediate files (dry-run default) |

## Requirements

- **GPU**: NVIDIA RTX 3090/4090 or higher (24GB VRAM recommended)
- **CUDA**: 11.8 or 12.1
- **Python**: 3.10+
- **ComfyUI**: Latest stable
- **OS**: Linux (RunPod / Ubuntu 22.04)
- **ffmpeg**: Required for video processing

## Workflows

Six pre-built workflows are included:

| Workflow | Description |
|----------|-------------|
| `01_quick_test.json` | Minimal 2-node test |
| `02_calibration_test.json` | Test preprocessing chain |
| `03_masking_test.json` | Test through quality gate |
| `04_training_preview.json` | Quick 200-iteration training test |
| `05_full_pipeline.json` | Complete 14-node production pipeline |
| `06_inference_only.json` | Render from pre-trained model |

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](docs/INSTALL.md) | Installation guide |
| [TROUBLESHOOT.md](docs/TROUBLESHOOT.md) | Common issues and fixes |
| [PIPELINE_EXPLAINED.md](docs/PIPELINE_EXPLAINED.md) | Stage-by-stage pipeline breakdown |
| [NODES_REFERENCE.md](docs/NODES_REFERENCE.md) | Complete node API reference |
| [RUNPOD_GUIDE.md](docs/RUNPOD_GUIDE.md) | RunPod setup and optimization |
| [NLE_EXPORT_GUIDE.md](docs/NLE_EXPORT_GUIDE.md) | Premiere Pro / Resolve export |
| [CAMERA_RIG_GUIDE.md](docs/CAMERA_RIG_GUIDE.md) | Multi-camera capture best practices |

## Architecture

The pipeline flows a `DATASET_INFO` typed dictionary through all nodes. Each node reads required fields, performs its operation, and returns an updated copy. This design ensures:

- **No crashes** — `BaseEasyVolcapNode._safe_execute()` wraps all logic
- **Immutable data flow** — each node returns a new dict, never mutates
- **Checkpoint/resume** — sentinel files track completed operations
- **Quality enforcement** — Quality Gate blocks training on bad data

## Credits

- [4K4D](https://zju3dv.github.io/4k4d/) — Zhu et al., CVPR 2024
- [EasyVolcap](https://github.com/zju3dv/EasyVolcap) — Zhejiang University
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — comfyanonymous
- [COLMAP](https://colmap.github.io/) — Schönberger & Frahm
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) — Lin et al.

## License

MIT License — see [LICENSE](LICENSE) for details.

**Note**: The underlying 4K4D model weights and EasyVolcap framework may have separate license terms (non-commercial research). Check their repositories for details before commercial use.

# ComfyUI-4K4D Installation Guide

## Prerequisites

- **GPU**: NVIDIA RTX 3090/4090 or higher (24GB VRAM recommended)
- **CUDA**: 11.8 or 12.1
- **Python**: 3.10+
- **ComfyUI**: Latest stable release
- **ffmpeg**: Required for video processing
- **COLMAP**: Required for camera calibration (auto-installed by Node 2)

## Quick Install (RunPod)

1. Start a RunPod instance using the **ComfyUI Docker template** with an **RTX 4090**.

2. Clone into ComfyUI's custom nodes directory:
   ```bash
   cd /workspace/ComfyUI/custom_nodes
   git clone https://github.com/Kariembassam/Claude_4K4D.git ComfyUI-4K4D
   ```

3. Install Python dependencies:
   ```bash
   cd ComfyUI-4K4D
   pip install -r requirements.txt
   ```

4. Restart ComfyUI. The 14 4K4D nodes will appear under the **4K4D/** category.

5. In your workflow, connect a **DependencyInstall** node first. It handles:
   - PyTorch verification
   - PyTorch3D compilation
   - EasyVolcap + 4K4D clone and install
   - CUDA extensions (diff-point-rasterization, tinycudann)
   - COLMAP installation
   - RobustVideoMatting download

## Manual Install (Local Machine)

1. Clone the repository:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/Kariembassam/Claude_4K4D.git ComfyUI-4K4D
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-4K4D
   pip install -r requirements.txt
   ```

3. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg colmap

   # macOS (CPU only — no training support)
   brew install ffmpeg colmap
   ```

4. Restart ComfyUI.

## Verifying Installation

After restarting ComfyUI:

1. Open the node menu (right-click on canvas)
2. Navigate to **4K4D/** category
3. You should see 14 nodes across subcategories:
   - 4K4D/Input
   - 4K4D/Setup
   - 4K4D/Preprocessing
   - 4K4D/Processing
   - 4K4D/Training
   - 4K4D/Rendering
   - 4K4D/Output
   - 4K4D/Utilities

## Dependency Install Node

The **DependencyInstall** node (Node 2) automates the entire environment setup. It runs a 15-step installation sequence and uses sentinel files to skip completed steps on subsequent runs.

### Install Modes

| Mode | Description |
|------|-------------|
| `stable` | Uses pinned versions tested on RTX 4090 |
| `latest` | Uses latest releases (may have compatibility issues) |

### What Gets Installed

1. PyTorch verification/installation
2. PyTorch3D from source
3. EasyVolcap repository clone + install
4. 4K4D repository clone + install
5. diff-point-rasterization (CUDA extension)
6. tinycudann (CUDA extension)
7. COLMAP (if not present)
8. RobustVideoMatting model weights

## Troubleshooting

See [TROUBLESHOOT.md](TROUBLESHOOT.md) for common issues.

## CUDA Architecture

The node pack auto-detects your GPU's CUDA compute capability. Supported architectures:

| GPU | Compute Capability |
|-----|-------------------|
| RTX 3090 | 8.6 |
| RTX 4090 | 8.9 |
| A100 | 8.0 |
| H100 | 9.0 |

If auto-detection fails, set `TORCH_CUDA_ARCH_LIST` manually:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

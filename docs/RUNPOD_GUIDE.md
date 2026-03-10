# RunPod Setup Guide for ComfyUI-4K4D

## Recommended Configuration

| Setting | Value |
|---------|-------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| Template | ComfyUI Docker |
| Disk | 100GB+ (training data + checkpoints) |
| Volume | Mount to /workspace |

## Step-by-Step Setup

### 1. Create Pod

1. Go to RunPod and select "Secure Cloud" or "Community Cloud"
2. Choose an RTX 4090 instance
3. Select the **ComfyUI** template (or any PyTorch 2.x template)
4. Set disk size to at least 100GB
5. Deploy the pod

### 2. Install ComfyUI-4K4D

```bash
# Connect via web terminal or SSH
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/Kariembassam/Claude_4K4D.git ComfyUI-4K4D
cd ComfyUI-4K4D
pip install -r requirements.txt
```

### 3. Upload Your Data

Upload your multi-camera footage to the pod:

```bash
# Option 1: SCP from local machine
scp -r ./my_scene/ runpod:/workspace/data/my_scene/

# Option 2: wget from cloud storage
mkdir -p /workspace/data/my_scene
cd /workspace/data/my_scene
wget https://your-storage.com/cam00.mp4
wget https://your-storage.com/cam01.mp4
# ... etc
```

### 4. Restart ComfyUI

```bash
# Kill existing ComfyUI process
pkill -f "python.*main.py" || true

# Restart with GPU support
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### 5. Access ComfyUI

Open the RunPod proxy URL (usually `https://xxxx-8188.proxy.runpod.net/`)

### 6. Load a Workflow

1. In ComfyUI, click "Load" in the menu
2. Navigate to the workflows/ directory
3. Start with `01_quick_test.json` to verify everything works
4. Progress to `05_full_pipeline.json` for production runs

## Directory Structure on RunPod

```
/workspace/
├── ComfyUI/
│   ├── custom_nodes/
│   │   └── ComfyUI-4K4D/     ← The node pack
│   ├── models/
│   └── main.py
└── data/
    └── my_scene/              ← Your multi-camera footage
        ├── cam00.mp4
        ├── cam01.mp4
        └── ...
```

## Storage Management

RunPod charges for disk storage. Tips to manage space:

1. **Use the StatusMonitor node** to track disk usage in real-time
2. **Use the Cleanup node** with `dry_run=True` first to preview what will be deleted
3. **Delete intermediate files** after successful training (extracted frames, COLMAP database)
4. **Archive important results** to cloud storage before terminating the pod

### Typical Space Usage

| Item | Size |
|------|------|
| Input videos (10 cameras, 10s, 1080p) | ~2GB |
| Extracted frames | ~5GB |
| COLMAP database | ~1GB |
| Masks | ~500MB |
| Training checkpoints | ~10-20GB |
| Rendered output | ~1-2GB |
| **Total** | **~20-30GB** |

## Performance Expectations

| Operation | RTX 4090 Time |
|-----------|---------------|
| Dependency install (first run) | 15-30 min |
| Frame extraction (10 cameras, 300 frames) | 2-5 min |
| COLMAP calibration | 10-30 min |
| Mask generation (RVM) | 5-10 min |
| Visual hull | 1-2 min |
| Training (preview_static, 200 iter) | 10 min |
| Training (full_sequence, 1600 iter) | 12-24 hours |
| SuperCharge conversion | 5-10 min |
| Rendering (120 frames, spiral) | 2-5 min |

## Tips for RunPod

1. **Sentinel files persist** — The DependencyInstall node skips completed steps, so restarting the pod and re-running is fast
2. **Training auto-resumes** — If your pod restarts mid-training, the Train node picks up from the last checkpoint
3. **Use preview_static first** — Always validate with a quick 200-iteration run before committing to full training
4. **Monitor VRAM** — The StatusMonitor node shows GPU memory usage; if you hit OOM, reduce resolution_scale

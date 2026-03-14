# ComfyUI-4K4D Troubleshooting Guide

## Common Issues

### Node Not Appearing in ComfyUI

**Symptom**: 4K4D nodes don't show up in the node menu.

**Solutions**:
1. Check ComfyUI console for import errors
2. Verify `requirements.txt` dependencies are installed:
   ```bash
   pip install jinja2 pyyaml psutil pillow numpy
   ```
3. Ensure the directory is in `custom_nodes/`:
   ```
   ComfyUI/custom_nodes/ComfyUI-4K4D/__init__.py
   ```
4. Restart ComfyUI completely (not just refresh)

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use `preview_static` training mode first (lower VRAM usage)
2. Reduce `resolution_scale` from 1.0 to 0.5 in training parameters
3. Close other GPU processes: `nvidia-smi` to check
4. On RTX 4090 (24GB), full training supports up to ~20 cameras at 1080p

### PyTorch3D Build Failure

**Symptom**: `error: command 'nvcc' failed`

**Solutions**:
1. Verify CUDA toolkit is installed: `nvcc --version`
2. Set CUDA architecture explicitly:
   ```bash
   export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090
   ```
3. Ensure GCC version compatibility (GCC 11 recommended)
4. Try the DependencyInstall node again — it auto-retries

### COLMAP Not Found

**Symptom**: Camera calibration fails with "colmap not found"

**Solutions**:
1. Let the DependencyInstall node handle it (installs automatically)
2. Manual install:
   ```bash
   # Ubuntu
   sudo apt-get install colmap

   # From source
   git clone https://github.com/colmap/colmap
   cd colmap && mkdir build && cd build
   cmake .. && make -j$(nproc) && sudo make install
   ```
3. Verify: `colmap -h`

### ffmpeg/ffprobe Not Found

**Symptom**: Frame extraction fails with "ffprobe not found"

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# RunPod (usually pre-installed)
which ffmpeg ffprobe
```

### Quality Gate Blocking Training

**Symptom**: Training node refuses to start, says quality gate not passed.

**This is intentional!** The Quality Gate (Node 6b) must pass before training. Check the quality report for what failed:

1. **Mask quality**: Re-run mask generation, or check mask images manually
2. **Blur/sharpness**: Remove blurry frames from source footage
3. **Sync alignment**: Re-sync videos or use `method=none` if already synced
4. **Camera coverage**: Add more cameras (info only, doesn't block)

To force-bypass: Set `force_pass=True` and type "I UNDERSTAND" in the confirmation field. Training may produce poor results.

### Training PSNR Too Low

**Symptom**: PSNR < 20dB at iteration 100

**Solutions**:
1. Check camera calibration quality — re-run COLMAP with more features
2. Verify masks are correct — bad masks cause training artifacts
3. Ensure scene is within bounding box — check visual hull output
4. Try with fewer cameras first to validate pipeline

### Import Errors: jinja2 / yaml / psutil

**Symptom**: `ModuleNotFoundError: No module named 'jinja2'`

**Solution**:
```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-4K4D
pip install -r requirements.txt
```

### RunPod-Specific Issues

#### Disk Space
RunPod instances have limited disk. Monitor usage:
- Use the StatusMonitor node to check disk
- Use the Cleanup node to remove intermediate files
- Training data + checkpoints can use 50GB+

#### Persistent Storage
- Store datasets in `/workspace/` (persists across pod restarts)
- The active ComfyUI install path varies by RunPod template — check which one is running (e.g. `/workspace/runpod-slim/ComfyUI/` vs `/workspace/ComfyUI/`)
- Custom node directory persists under whichever ComfyUI install is active

#### Multiple ComfyUI Installs
Some RunPod templates have two ComfyUI installs (e.g. `/workspace/ComfyUI/` and `/workspace/runpod-slim/ComfyUI/`). Only one is active. Symptoms of using the wrong path:
- Training completes but no checkpoint files appear
- Data directories don't match between nodes
- Symlinks pointing to the wrong install cause silent path mismatches

**Solution**: The nodes use relative paths from their own install location. Ensure you cloned into the **active** ComfyUI's `custom_nodes/` directory. Check which ComfyUI is running: `ps aux | grep main.py`

#### GPU Not Detected
```bash
nvidia-smi  # Should show GPU info
python -c "import torch; print(torch.cuda.is_available())"
```

### Training Crashes with IndexError: list index out of range

**Symptom**: `IndexError: list index out of range` at `self.frame_sample[1]`

**Cause**: The `frame_sample_range` widget was set to "none" or an invalid value. EasyVolcap expects a 3-element list `[start, end, step]`.

**Solution**: Leave `frame_sample_range` as "none" (the node now correctly skips it) or provide a valid range like "0,24,1". This was fixed in commit `364aa75`.

### Viewer Not Rendering

**Symptom**: WebGL viewer shows blank canvas

**Solutions**:
1. The bundled Three.js r128 library should work out of the box. If the 3D View tab is blank, check the browser console for JS errors
2. Video tab should work regardless — check that render output files exist
3. Browser must support WebGL2: check at `chrome://gpu`
4. Ensure PLY files were generated — check the render output `ply/` directory

## Getting Help

1. Check ComfyUI console output for detailed error logs
2. Look in the `logs/` directory for per-node log files
3. Open an issue on GitHub with:
   - GPU model and VRAM
   - CUDA version (`nvcc --version`)
   - Python version
   - Full error traceback
   - ComfyUI version

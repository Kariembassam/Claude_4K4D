# Camera Rig Guide for 4K4D Capture

Best practices for multi-camera capture setups compatible with the 4K4D pipeline.

## Minimum Requirements

| Parameter | Minimum | Recommended |
|-----------|---------|-------------|
| Cameras | 4 | 8-20 |
| Resolution | 720p | 1080p |
| Framerate | 24 fps | 30 fps |
| Duration | 2 seconds | 5-15 seconds |
| Codec | Any (auto-transcoded) | H.264 |

## Camera Placement

### Hemisphere Layout (Recommended)
- Place cameras in a hemisphere around the subject
- Maintain roughly equal angular spacing
- Ensure at least 60% overlap between adjacent camera views
- Avoid placing all cameras at the same height

### Ring Layout
- Cameras in a single ring at subject height
- Good for bust/portrait captures
- Limited vertical coverage (no top-down views)

### Sparse Layout
- 4-6 cameras at key viewpoints
- Relies more heavily on neural interpolation
- Lower quality but more practical for mobile setups

## Camera Settings

### Synchronization
The pipeline supports three sync methods:

1. **Hardware sync** (best): Genlock or trigger cable between cameras
2. **Audio sync** (good): Clap or audio tone — the pipeline auto-syncs via cross-correlation
3. **No sync** (acceptable): Manual alignment — works for static scenes only

### Exposure
- Use **manual exposure** — auto exposure causes flickering between cameras
- Match exposure settings across all cameras
- Slight variations are tolerable (the model adapts)

### Focus
- Use **manual focus** — auto focus hunting ruins frames
- Set focus to the subject distance
- Use a higher f-stop (f/5.6-f/8) for deeper depth of field

### White Balance
- Use **manual white balance** — should be identical across all cameras
- Set a custom white balance using a gray card

## Supported Camera Types

The pipeline is format-agnostic. Tested configurations:

| Camera Type | Notes |
|-------------|-------|
| GoPro Hero 10/11/12 | Excellent. Use linear lens mode, ProTune ON |
| Sony Alpha (mirrorless) | Excellent. Use S-Log3 for maximum dynamic range |
| iPhone 14/15 Pro | Good. Use Filmic Pro for manual controls |
| Blackmagic Pocket | Excellent. Native ProRes support |
| Intel RealSense | Supported. RGB stream only (depth not used) |
| Webcams | Basic. Low quality limits reconstruction |

## Scene Considerations

### Good Scenes
- Single person performing
- Object on turntable
- Dance/movement in bounded area
- Sports action in defined zone

### Challenging Scenes
- Transparent or reflective objects
- Very thin structures (hair, lace)
- Extremely fast motion (motion blur)
- Large unbounded environments

### Lighting
- Even, diffuse lighting is best
- Avoid harsh shadows from single-point sources
- Avoid mixed color temperatures
- Studio softbox lighting is ideal

## File Organization

Before ingesting into the pipeline, organize your footage:

```
my_scene/
├── cam00.mp4    (Camera 0 - front)
├── cam01.mp4    (Camera 1 - front-left)
├── cam02.mp4    (Camera 2 - left)
├── cam03.mp4    (Camera 3 - back-left)
├── cam04.mp4    (Camera 4 - back)
├── cam05.mp4    (Camera 5 - back-right)
├── cam06.mp4    (Camera 6 - right)
├── cam07.mp4    (Camera 7 - front-right)
└── ...
```

### Naming Convention
- Any consistent naming works: `cam00.mp4`, `camera_1.mov`, `GOPR0001.MP4`
- The FolderIngest node detects video files regardless of naming
- Camera order is determined by alphabetical sorting of filenames

## Pre-existing Calibration

If you have COLMAP or other calibration data:

1. Place `extri.yml` and `intri.yml` in the dataset root
2. Set CameraCalibration node to `method=existing`
3. The pipeline will use your calibration instead of running COLMAP

### EasyVolcap Calibration Format

```yaml
# extri.yml
cam00:
  R: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
  T: [tx, ty, tz]

# intri.yml
cam00:
  K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  dist: [k1, k2, p1, p2, k3]
```

## Troubleshooting Capture Issues

| Issue | Solution |
|-------|----------|
| Flickering between cameras | Use manual exposure and white balance |
| Blurry frames | Increase shutter speed, improve lighting |
| COLMAP fails | Add more cameras, improve scene texture |
| Poor masks | Improve background contrast, add green screen |
| Sync drift | Use shorter clips, add audio sync markers |

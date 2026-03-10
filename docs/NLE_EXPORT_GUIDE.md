# NLE Export Guide

Export 4K4D renders for professional video editing in Adobe Premiere Pro and DaVinci Resolve.

## Export Node (Node 12)

The ExportPack node packages your rendered output into NLE-ready formats.

### Target NLE Options

| Target | Output Format | Description |
|--------|--------------|-------------|
| `premiere` | ProRes 4444 + XML | Optimized for Adobe Premiere Pro |
| `resolve` | ProRes 4444 + EDL | Optimized for DaVinci Resolve |

## Premiere Pro Workflow

### 1. Generate Export Pack

In ComfyUI, connect the ExportPack node after Render:
- Set `target_nle` to `premiere`
- Enable `include_proxies` for smoother editing

### 2. Import into Premiere

1. Open Premiere Pro
2. File → Import → select the `.xml` project file from the export pack
3. The project will contain:
   - Full-resolution ProRes 4444 renders
   - Proxy media (if generated)
   - Organized bin structure

### 3. Working with Proxies

If proxies were generated:
1. Toggle proxy editing: Button in Program Monitor toolbar
2. Full-res clips are used for final export
3. Proxies are used during editing for performance

## DaVinci Resolve Workflow

### 1. Generate Export Pack

- Set `target_nle` to `resolve`
- Enable `include_proxies` if needed

### 2. Import into Resolve

1. Open DaVinci Resolve
2. Media Pool → right-click → Import Media
3. Or use File → Import → Timeline (EDL)
4. Select the `.edl` file from the export pack

### 3. Color Grading

4K4D renders are in linear color space. In Resolve:
1. Set project color science to DaVinci YRGB Color Managed
2. Set timeline color space to Rec.709
3. Apply a linear-to-Rec.709 transform if needed

## Export Pack Contents

The export pack directory contains:

```
export_pack_<timestamp>/
├── renders/
│   ├── spiral_render.mov          (ProRes 4444, full-res)
│   ├── spiral_render_proxy.mp4    (H.264, proxy)
│   └── frames/                    (PNG sequence if requested)
├── project/
│   ├── project.xml                (Premiere) or project.edl (Resolve)
│   └── metadata.json              (Pipeline metadata)
├── camera_data/
│   ├── extri.yml                  (Camera extrinsics)
│   └── intri.yml                  (Camera intrinsics)
└── README.txt                     (Quick reference)
```

## Output Format Details

### ProRes 4444
- Codec: Apple ProRes 4444
- Color depth: 10-bit
- Alpha: Yes (if masks available)
- Quality: Production grade, minimal compression artifacts
- File size: ~1GB per minute at 1080p

### H.264 (Web/Preview)
- Codec: H.264 High Profile
- Color depth: 8-bit
- CRF: 18 (high quality)
- File size: ~50MB per minute at 1080p

### PNG Sequence
- Format: 16-bit PNG
- One file per frame
- Maximum quality, no compression artifacts
- Useful for compositing in After Effects / Nuke

## Tips

1. **Always render ProRes for NLE** — H.264 causes decode lag on the timeline
2. **Use PNG sequences** for After Effects / Nuke compositing
3. **Include proxies** if editing on a laptop or slower machine
4. **Camera data is included** for VFX integration (match-moving)

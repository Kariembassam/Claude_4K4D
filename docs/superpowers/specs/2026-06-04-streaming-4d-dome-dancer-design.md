# Streaming-4D for dome_dancer — Design Spec

- **Date:** 2026-06-04
- **Status:** Design approved. Implementation gated on the floater-fix validation (step 1).
- **Approach chosen:** A — sequential warm-started per-frame fits ("streaming 4D").

## Context

The `dome_dancer` dataset is a 24-camera geodesic capture dome around a dancing humanoid
rig (131 frames, 4K, ground-truth `extri.yml`/`intri.yml`, empty `points3D` → vhull init).
Prior results on this data:

- **Dynamic 4K4D model** (`r4dv`, one 262 144-point / 64³ budget shared across all 131
  frames), trained to ~32 000 iters: per-frame render caps at **PSNR ≈ 23**, dancer faint and
  buried in floaters. More training gave diminishing returns (≈2 dB over 5× iters).
- **Silhouette-culled splat export** (≈80 k pts/frame): removes floaters but is a **puffy,
  sparse blob** from novel views — not v5-level.
- **Single-frame static fit** (frame 65, ~4 000 iters): **PSNR ≈ 35**, clean recognizable
  dancer body — *above* v5's ~30. Confirms the data/calibration are good and that per-frame
  capacity is the bottleneck. Residual black floaters persist (a separate issue).

**v5** = the earlier sharp static reconstruction (PSNR ~30, SSIM ~0.98) used as the quality bar.

## Goal

A 131-frame 4D splat where each frame is **v5-level sharp**, playback is **temporally
coherent** (no jitter), and the result is **floater-free**.

## Key insight

The static fit shows the model's **per-point alpha** concentrates on the true surface (sharp
body), while floaters are low-alpha off-surface points. **Alpha is the lever for both
sharpness and floater removal**: keep only high-alpha (confident-surface) points. Silhouette
culling alone removes floaters but yields puffy geometry; alpha is what makes it sharp.

## Pipeline

1. **Floater / sharpness fix — GATING, validate on frame 65 first (~minutes).**
   Extract the geo-regressor's per-point alpha and keep only confident-surface points
   (threshold τ). Try cheapest viable route in order:
   (a) render-time alpha threshold hook in the `r4dv` sampler render path;
   (b) post-hoc alpha export (run the sampler geometry forward, keep alpha > τ);
   (c) `immask_crop=True` retrain (v5 may have used this).
   Pick whichever gives a **clean AND sharp** frame-65. **Fallback:** silhouette-cull (clean
   but puffy). **If none yields clean+sharp → STOP and rethink before any long run.**
2. **Frame-0 anchor fit:** full ~4 000-iter static fit (cached vhull → `reload_vhulls=False`,
   `PYOPENGL_PLATFORM=egl`).
3. **Sequential warm-start loop (frames 1 → 130):** each frame initialises from the previous
   frame's checkpoint, `frame_sample=[t,t+1,1]`, refines ~600–800 iters. Consecutive poses are
   similar → coherent evolution (low jitter) + fast convergence. Per-frame checkpoint →
   resumable. **Tune the refine-iter knee on frames 1–3 and report before committing all 131.**
4. **Export + assemble:** alpha-clean per-frame PLY → 131-frame splat sequence for the Viewer
   node + an orbit/turntable video.
5. **Validate:** camera-0 + novel-view turntable at frames 0 / 65 / 130 vs GT; confirm
   v5-level per frame + smooth playback.

## Risks & mitigations

- **Alpha extraction harder than expected** → ordered fallbacks in step 1; silhouette-cull as
  floor.
- **Warm-start iters mis-tuned** (too few = blur, too many = slow) → tune on frames 1–3 first.
- **Compute ~5–10 h** → run in background with per-frame checkpoints (resumable / interruptible).
- **Residual temporal jitter** → optional light temporal smoothing of point positions.

## Testing

- Per-frame PSNR (target ≥ 30).
- Visual spot-checks at frames 0 / 65 / 130 (training camera + novel turntable view).
- Playback smoothness check across the assembled sequence.

## Environment notes (pod `e1yl08p60futpy`)

The `/workspace` network volume persists across pod swaps; the container root does **not**.
On a fresh pod, restore before running:

- SSH authorized key (user-added).
- `/usr/local/bin/python` wrapper → `.venv-cu128/bin/python` (EasyVolcap shells out to bare
  `python`).
- `libEGL.so.1` (`apt-get install libegl1 libglvnd0 libgles2`) + `PYOPENGL_PLATFORM=egl`.

Rendering uses CUDA (diff-point-rasterization); only the vhull **carve** needs OpenGL/EGL.
Cached vhulls at `data/dome_dancer/vhulls/` allow `reload_vhulls=False` to skip the carve.

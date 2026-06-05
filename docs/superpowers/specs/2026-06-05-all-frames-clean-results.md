# All-frames-clean retrain — results + implementation notes

- **Date:** 2026-06-05
- **Outcome:** SUCCESS. Every frame of the dynamic `full_sequence` model is now uniformly
  clean (not just frame 65).

## The diagnosis (why only frame 65 was clean)

Frame 65 had received a **dedicated 4000-iter static fit** (separate model
`4k4d_dome_f65_static`, PSNR ~35). The dynamic `full_sequence` model had ~32 000 iters spread
across **131 frames ≈ ~244 iters/frame** — ~16× less per-frame training — so all frames
(65 included, in the dynamic model) were undertrained noise-clouds (~PSNR 23).

**Per-frame iteration budget is the lever.** To give every frame ~K dedicated iters in the
dynamic model: total iters ≈ K × 131 (each train step touches one (view, frame) sample).

## The fix (brute-force retrain — user's call: shorter-first to de-risk)

Resumed `full_sequence` to **~1500 iters/frame** = epochs=123 × ep_iter=1600 ≈ **196 800 iters**
(~15 h on the RTX 5090, ~3 iters/s), same config as the 32k run, eval suppressed
(`eval_ep=99999`), per-epoch `latest.pt` to the `/workspace` volume (restart-resilient).

**Result (camera-0 eval, 131 frames): PSNR mean 29.58, min 27.14, max 31.35** — uniform, no
outliers; clean recognizable dancer at every frame (vs ~23 noise before). At v5 level (~30).
Residual light surface speckle remains = the shared-network ceiling vs a dedicated fit's 35
(the per-frame point clouds give per-frame geometry, but the geo/IBR networks are shared).
Optional future: push to ~4000 iters/frame (~+30 h) to shave it toward 35.

## Key fixes this phase

- **CUDA OOM** at the 1080×1920 LPIPS step: a GPU-resident ComfyUI (`main.py --listen`, ~0.5 GB)
  tipped the peak over 31 GB. Fix: kill it (frees the full 32 GB) + `export
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Fresh-pod env restore** (only `/workspace` persists): recreate `/usr/local/bin/python` →
  `.venv-cu128/bin/python`; `apt-get install libegl1 libglvnd0 libgles2` + `PYOPENGL_PLATFORM=egl`;
  user re-adds the SSH key.

## Streaming interactive viewer (Phase 1 — working)

- **Server (pod):** `/workspace/serve.sh` — EasyVolcap GUI runner with `configs/specs/server.yaml`
  (`viewer_cfg.type=WebSocketServer`), `viewer_cfg.host=127.0.0.1 port=1024
  window_size=[800,800] model_cfg.sampler_cfg.bg_brightness=1.0` (white bg). Patched
  `easyvolcap/runners/websocket_server.py` for **websockets-16** (`async with websockets.serve(...)`
  + `asyncio.run`; handler `path=None`).
- **Transport:** `ssh -N -L 1024:127.0.0.1:1024` from the user's Mac (browser → `ws://localhost:1024`).
- **Client:** `web/viewer4d_stream.html` — canvas draws streamed JPEG, sends
  `zlib(JSON camera)` per render; **30 fps wall-clock playback timer** (decoupled from render fps);
  orbit calibrated to the capture dome (look-at `[0,0,0.6]`, radius 3.33, elev ~33°, world_up
  `[0,0,1]`); served locally via `python -m http.server` (http origin, so `ws://` isn't blocked).

## Other pod-side patches (from streaming exploration — not needed for the brute-force retrain)

- `easyvolcap/models/samplers/point_planes_sampler.py` `_load_state_dict_pre_hook`: non-strict
  per-frame pcd load (skip frames absent from checkpoint, keep vhull-init) + `runner_cfg.strict=False`
  — enables growing-frame resume. Single-growing-model streaming is brittle (also fights optimizer
  param-group counts); the brute-force retrain avoided all of it.
- `easyvolcap/models/samplers/r4dv_sampler.py`: env-gated `DUMP_FRAGS` hook to dump per-point
  xyz/rgb/rad/occ (debug). `occ` is ~1 for ~180k points incl. floaters → can't threshold floaters by occ.

## Next (Phase 2)

Integrate the streaming viewer into the ComfyUI **Viewer node** as a tab so everything runs in
ComfyUI (likely a ComfyUI aiohttp ws-proxy route → `ws://localhost:1024`, + a node-embedded canvas).

# Live 4D video exporter — design

- **Date:** 2026-06-05
- **Status:** Approved (1080p, download + Video tab).

## Goal

In the Live 4D tab, let the user orbit to frame a shot, then one-click **export a
high-quality MP4 of the full 131-frame performance rendered from that exact camera angle**
— the rasterized-clean 4K4D render (not the point cloud). For previz reference / editing.

## Architecture

- **3D camera = the Live 4D orbit.** No separate camera object (v1). The current
  az/el/radius is the export camera.
- **Frontend (`fourk4d_viewer.js`, Live 4D tab):** an **"Export MP4"** button. On click it
  builds the current camera at the export resolution (R, T from the orbit; K for 1080;
  tight `bounds`) and `POST`s it to `/4k4d/export_video`. Then it polls the returned log
  (via `/4k4d/view`) showing "Rendering N/131…", and on `EXPORT DONE` triggers a browser
  **download** of the MP4 and loads it into the Video tab.
- **Backend (`n10_viewer.py`): new `POST /4k4d/export_video` route.** Reads
  `{R,T,K,bounds,H,W,nframes}`, kicks off a background `asyncio` task (returns immediately
  with `{mp4_url, log_url}` to avoid request timeout), which: connects to the render-server
  (`ws://127.0.0.1:1024`), renders each frame at the fixed camera varying only time, **holding
  each camera until the returned JPEG bytes actually change** before saving (the server's
  render loop is decoupled from the socket and runs slower at 1080, so a fixed "settle" count
  drains stale frames faster than the GPU makes fresh ones → only ~5 unique frames out of 131
  → choppy; encoding is deterministic, so waiting for a byte change yields one genuinely fresh
  render per timestep = 131 distinct frames), saves the **raw JPEGs**, then `ffmpeg -vf vflip` (the server render is
  flipped) → H.264 MP4 under the dataset `render/` dir. Progress + `EXPORT DONE` written to
  the log.
- **Render-server:** bumped to `window_size=[1080,1080]` so both Live 4D and the export are
  1080p (export ~1–2 min at ~3 fps; Live 4D buffer ~40 s).

## Data flow

orbit (client) → POST camera → background render task ↔ render-server (ws) → JPEGs → ffmpeg →
MP4 → client polls log → download + Video tab.

## Error handling

- Render-server down → route logs `EXPORT ERROR`; client surfaces it.
- ffmpeg missing/fail → `EXPORT FAIL`.
- Long render → non-blocking (background task + log polling), so no proxy/request timeout.

## Out of scope (v1)

Saveable named cameras, multi-angle batch, Blender-native asset (the point-cloud Alembic
route is parked — superseded by this rasterized-video exporter for previz).

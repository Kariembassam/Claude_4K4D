# Live 4D camera refinement + keyframe timeline — design

- **Date:** 2026-06-05
- **Status:** Approved (Catmull-Rom spline interpolation; right-drag pan deferred to v2).

## Goal

Refine the ComfyUI-4K4D Viewer node's **Live 4D** tab: (1) **smoother camera controls**,
(2) a **camera keyframe timeline synced to the 131-frame performance**, and (3) a **cleaner
layout**. Keyframed camera moves play in the live preview **and** are rendered by the MP4 export,
so the exported clip matches what you previewed.

## Interaction model — the core change

Today every orbit/zoom calls `invalidate()` → re-buffers all 131 frames (~20 s) → janky. Split
the viewer into two modes:

- **Interact** (orbit / zoom / scrub): render only the **current** frame live via the ws
  ping-pong (~6 fps), camera **eased**. No re-buffer.
- **Play/Preview**: build the 131-frame cache once, each frame `i` rendered at `cameraAt(i)`;
  play at a true 30 fps. Re-buffer only when you press Play after editing keyframes.

## Components

### 1. Smooth camera (client — `web/js/fourk4d_viewer.js`)
- State = current `{az,el,radius}` + **target** `{az,el,radius}`. Pointer-drag / wheel update the
  *targets*; a `requestAnimationFrame` loop eases current → target (`cur += (tgt-cur)*k`, k≈0.18;
  radius eased in log space). Produces glide after release, no snapping.
- While interacting (or paused) continuously send the eased camera at the current playhead frame
  and draw the returned JPEG (the proven `viewer4d_stream` ping-pong).
- Tunable orbit (rad/px) and zoom sensitivity constants. **Pan: deferred (v2).**

### 2. Keyframe timeline synced to the dance (client)
- Data = `keys`: sorted array of `{f: 0..130, az, el, radius}`.
- **◆ Add Key**: capture the eased orbit at the current playhead `f` → insert/replace key at `f`.
  **✕ Del Key**: remove the selected (or nearest) key.
- `cameraAt(f)`: `keys.length < 2` → that single key's orbit (or the current orbit if none).
  Else **Catmull-Rom spline** through `(az*, el, radius)` vs `f` — `az*` is az **unwrapped**
  (accumulate shortest-path deltas) so interpolation never jumps; `el` clamped `[-1.45, 1.45]`,
  `radius` clamped `[2, 8]`; endpoints use duplicated tangents (hold beyond first/last key).
- **Timeline UI**: a track (0..130) under the canvas with a playhead + ◆ markers at key frames.
  Click track → scrub (set `f`, interact-render `cameraAt(f)`). Selecting a marker highlights it
  for Del. (Marker drag-to-retime deferred to v2 — retime via del + re-add.)

### 3. Keyframed preview (client)
- **Play** rebuilds the cache where frame `i` is rendered at `cameraAt(i)` (send camera+time per
  `i`, PRIME warm-up as today), then plays the cache at 30 fps. Same cache-then-play, per-frame
  camera. A new Play/interact cancels any in-flight buffer.

### 4. Export the move (server — `nodes/n10_viewer.py` `/4k4d/export_video`)
- Accept an optional `keyframes` array (`[{f,az,el,radius}]`) plus `center`, `world_up`,
  `bounds`, `K`, `H`, `W`, `nframes`.
- Port to Python: the orbit→extrinsics `buildCamera` and the `cameraAt` Catmull-Rom — **identical
  to the client**. Render frame `i` at `cameraAt(i)` via the existing `wait_fresh` capture (which
  already waits for a genuinely fresh render per timestep; now the camera varies per frame too).
- No `keyframes` → today's fixed-angle behavior (unchanged).

### 5. Layout (client + widget height)
Stacked panel instead of the cramped overlay:
```
canvas (~380px)
timeline track (~40px)   ◆──◆────────◆   playhead + key markers
controls row             ▶/Pause · 27/131 · ◆Add Key · ✕Del Key · ⌖ · ⤓Export
status line              live: playing 30fps
```
Bump the viewer widget `computeSize` height (~560px) so nothing overflows (the prior
button-visibility trap). Keep `stopPropagation` on canvas + timeline pointer events so LiteGraph
doesn't drag the node while interacting.

## Camera + interpolation contract (shared, must match JS ⇄ Python)

The exported move must equal the previewed move, so both sides implement one formula:

- `center C` (look-at; dancer body center), `world_up = [0,0,1]`.
- `eye = C + radius * [cos(el)cos(az), cos(el)sin(az), sin(el)]`.
- `fwd = normalize(C - eye)`; `right = normalize(fwd × up)`; `down = fwd × right`;
  `R = [right, down, fwd]` (rows, world→cam); `T = [-right·eye, -down·eye, -fwd·eye]`.
- **Catmull-Rom** (uniform), per channel `az*` / `el` / `radius`; tangents from neighbors,
  endpoints duplicated; sample at `u = (f - f0)/(f1 - f0)` within the segment bracketing `f`.

## Error handling
- 0 keys → static current angle (today's behavior); 1 key → static that angle.
- Server: malformed `keyframes` → fall back to the fixed `R,T` camera + log a warning.
- Starting a new buffer/interact cancels the previous in-flight one (no overlapping ping-pongs).

## Testing
- Orbit is fluid (eased) with **no mid-drag re-buffer** (interact renders a single frame).
- Place 2–3 keys at different frames/angles → scrub shows the interpolated camera → Play shows a
  smooth move synced to the dance → Export → MP4 has 131 unique frames showing the **same** move
  as the preview (spot-check a few frames against the live preview).
- Layout: all controls visible (verify bounding rects, as in the prior overlay fix).

## Out of scope (v1)
Right-drag pan, marker drag-to-retime, per-key easing curves, save/load camera paths, multiple
camera tracks.

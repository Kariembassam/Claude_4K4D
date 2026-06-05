"""
ComfyUI-4K4D Node 10: Viewer
===============================
4-tab in-browser viewer with embeddable iframe export.
Tab 1: Video player, Tab 2: WebGL orbit, Tab 3: Split-view, Tab 4: Iframe export.
"""

import base64
import logging
import os
from pathlib import Path

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import CATEGORIES, DATASET_INFO_TYPE

logger = logging.getLogger("4K4D.n10_viewer")

# Register custom API route to serve 4K4D files from absolute paths
# ComfyUI's built-in /view endpoint only serves from its own directories
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/4k4d/view")
    async def serve_4k4d_file(request):
        """Serve files from 4K4D data directories."""
        filepath = request.query.get("path", "")
        if not filepath or not os.path.isfile(filepath):
            return web.Response(status=404, text="File not found")
        return web.FileResponse(filepath)

    import asyncio
    import aiohttp

    @PromptServer.instance.routes.get("/4k4d/stream")
    async def fourk4d_stream_proxy(request):
        """Bridge a browser websocket <-> the EasyVolcap WebSocketServer render-server
        (default ws://127.0.0.1:1024) so the in-ComfyUI Live-4D viewer streams frames
        SAME-ORIGIN (via ComfyUI's own http/https host) — no separate port/tunnel and no
        mixed-content. Transparent byte pump: server sends JPEG, client sends zlib(JSON camera)."""
        target = request.query.get("target", "ws://127.0.0.1:1024")
        ws_client = web.WebSocketResponse(max_msg_size=0)
        await ws_client.prepare(request)

        async def pump(src, dst):
            async for m in src:
                if m.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(m.data)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(m.data)
                else:
                    break

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(target, max_msg_size=0) as ws_up:
                    await asyncio.gather(pump(ws_up, ws_client), pump(ws_client, ws_up))
        except Exception as e:
            logger.warning(f"4k4d stream proxy error ({target}): {e}")
            if not ws_client.closed:
                await ws_client.close()
        return ws_client

    # ---- Live-4D video exporter: render the full performance from the user's chosen
    #      orbit camera (rasterized-clean), encode an MP4, serve it for download ----
    import time as _time
    from urllib.parse import quote as _quote

    _EXPORT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "dome_dancer", "render")

    async def _do_export(cam, mp4, log_path, frames_dir, nframes):
        import os as _os, glob as _glob, json as _json, zlib as _zlib, subprocess as _sub, asyncio as _aio
        def logw(m):
            with open(log_path, "a") as f:
                f.write(m + "\n")
        _os.makedirs(frames_dir, exist_ok=True)
        for f in _glob.glob(frames_dir + "/*.jpg"):
            _os.remove(f)
        H, W, K, R, T, bounds = cam["H"], cam["W"], cam["K"], cam["R"], cam["T"], cam["bounds"]
        def camt(t):
            return {"H": H, "W": W, "K": K, "R": R, "T": T, "n": 0.02, "f": 100.0, "t": t, "v": 0.0,
                    "bounds": bounds, "mass": 0.1, "moment_of_inertia": 0.1, "movement_force": 1.0,
                    "movement_torque": 1.0, "movement_speed": 1.0, "origin": [0, 0, 0.13], "world_up": [0, 0, 1]}
        def cmsg(t):
            return _zlib.compress(_json.dumps(camt(t)).encode("ascii"))
        async def wait_fresh(ws, msg, prev, timeout=6.0):
            # The render-server runs a continuous render_loop (rasterizes self.camera, only
            # ~2-3 fps at 1080) decoupled from the server_loop (ships the LATEST rendered
            # image). So a frame received right after we change the time is usually a STALE
            # render of the previous time; blasting a fixed number of "settle" sends drains
            # stale frames faster than the GPU produces fresh ones -> duplicate frames ->
            # choppy video. Instead hold this exact camera+time and return the first frame
            # whose bytes DIFFER from prev (a genuinely fresh render of this time). Encoding
            # is deterministic, so identical renders compare equal. Timeout (e.g. a briefly
            # static pose) -> return the latest frame seen.
            t0 = _time.time(); data = prev
            while _time.time() - t0 < timeout:
                await ws.send_bytes(msg)
                m = await ws.receive()
                d = m.data
                if isinstance(d, (bytes, bytearray)):
                    if prev is None or d != prev:
                        return d
                    data = d
                await _aio.sleep(0.02)
            return data
        try:
            logw("rendering 0/%d" % nframes)
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect("ws://127.0.0.1:1024", max_msg_size=0) as ws:
                    m0 = await ws.receive()  # initial default-camera frame
                    prev = m0.data if isinstance(m0.data, (bytes, bytearray)) else None
                    for i in range(nframes):
                        # wait for a genuinely fresh render of this exact timestep (no dupes)
                        prev = await wait_fresh(ws, cmsg(i / (nframes - 1)), prev)
                        with open("%s/f%04d.jpg" % (frames_dir, i), "wb") as f:
                            f.write(prev)
                        if i % 10 == 0:
                            logw("rendering %d/%d" % (i, nframes))
            logw("rendering %d/%d" % (nframes, nframes))
            _os.makedirs(_os.path.dirname(mp4), exist_ok=True)
            _sub.run(["ffmpeg", "-y", "-framerate", "30", "-i", "%s/f%%04d.jpg" % frames_dir,
                      "-vf", "vflip", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=False)
            logw("EXPORT DONE" if _os.path.isfile(mp4) else "EXPORT FAIL")
        except Exception as e:
            logw("EXPORT ERROR " + str(e))

    @PromptServer.instance.routes.post("/4k4d/export_video")
    async def export_video(request):
        """Render the 131-frame performance at the posted (fixed) camera, varying time,
        through the render-server; encode an MP4. Non-blocking: returns URLs immediately,
        client polls the log."""
        body = await request.json()
        nframes = int(body.get("nframes", 131))
        cam = {"H": int(body["H"]), "W": int(body["W"]), "K": body["K"], "R": body["R"],
               "T": body["T"], "bounds": body["bounds"]}
        job = str(int(_time.time()))
        mp4 = os.path.abspath(os.path.join(_EXPORT_OUT, "export_%s.mp4" % job))
        log_path = "/tmp/4k4d_export_%s.log" % job
        frames_dir = "/tmp/4k4d_export_%s" % job
        open(log_path, "w").close()
        asyncio.create_task(_do_export(cam, mp4, log_path, frames_dir, nframes))
        return web.json_response({
            "mp4_path": mp4, "log_path": log_path,
            "mp4_url": "/4k4d/view?path=" + _quote(mp4),
            "log_url": "/4k4d/view?path=" + _quote(log_path),
        })

except Exception:
    pass  # Not running inside ComfyUI server context


class FourK4D_Viewer(BaseEasyVolcapNode):
    """
    4-tab viewer for 4K4D output visualization.

    Communicates with the frontend JavaScript (fourk4d_viewer.js)
    via PromptServer.send_sync to load data into the viewer widget.
    """

    CATEGORY = CATEGORIES["output"]
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("iframe_html_path", "iframe_embed_code")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "mp4_path": ("STRING", {"default": ""}),
                "ply_dir": ("STRING", {"default": ""}),
                "original_frames_dir": ("STRING", {"default": ""}),
                "default_tab": (["video", "webgl", "split", "iframe"], {"default": "video"}),
                "autoplay": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": True}),
                "iframe_title": ("STRING", {"default": "4K4D Preview"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, mp4_path="", ply_dir="", original_frames_dir="",
                default_tab="video", autoplay=True, loop=True,
                iframe_title="4K4D Preview", unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, mp4_path, ply_dir, original_frames_dir,
            default_tab, autoplay, loop, iframe_title, unique_id
        )

    def _run(self, dataset_info, mp4_path, ply_dir, original_frames_dir,
             default_tab, autoplay, loop, iframe_title, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root"])

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info.get("dataset_name", "output")

        # Auto-detect paths from dataset_info if not provided
        render_output = dataset_info.get("render_output", "")
        if not mp4_path and render_output:
            candidate = os.path.join(render_output, f"{name}_render.mp4")
            if os.path.exists(candidate):
                mp4_path = candidate

        if not original_frames_dir:
            original_frames_dir = os.path.join(dataset_root, "images", "00")

        # Encode small videos as base64 for reliable playback
        # (avoids path-serving issues entirely for typical preview renders)
        mp4_b64 = ""
        if mp4_path and os.path.exists(mp4_path):
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            if size_mb < 10:
                try:
                    with open(mp4_path, "rb") as f:
                        mp4_b64 = base64.b64encode(f.read()).decode()
                    self._node_logger.info(f"Encoded {size_mb:.1f}MB video as base64 for viewer")
                except Exception as e:
                    self._node_logger.warning(f"Failed to encode video as base64: {e}")

        # Auto-detect PLY directory from render output, dataset_info, or surfs/
        if not ply_dir:
            # Check render output PLY dir (populated by render node)
            ply_candidate = dataset_info.get("ply_dir", "")
            if ply_candidate and os.path.isdir(ply_candidate):
                ply_files = [f for f in os.listdir(ply_candidate) if f.endswith('.ply')]
                if ply_files:
                    ply_dir = ply_candidate
            if not ply_dir and render_output:
                ply_candidate = os.path.join(render_output, "ply")
                if os.path.isdir(ply_candidate):
                    ply_files = [f for f in os.listdir(ply_candidate) if f.endswith('.ply')]
                    if ply_files:
                        ply_dir = ply_candidate
            # Fallback: use surfs/ or vhulls/ from preprocessing (always valid)
            if not ply_dir:
                for subdir in ("surfs", "vhulls"):
                    candidate = os.path.join(dataset_root, subdir)
                    if os.path.isdir(candidate):
                        ply_files = [f for f in os.listdir(candidate) if f.endswith('.ply')]
                        if ply_files:
                            ply_dir = candidate
                            self._node_logger.info(f"Using {subdir}/ PLY files for 3D viewer")
                            break

        # Build list of PLY URLs servable via /4k4d/view endpoint
        ply_urls = []
        if ply_dir and os.path.isdir(ply_dir):
            ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])
            ply_urls = [
                f"/4k4d/view?path={os.path.join(ply_dir, f)}"
                for f in ply_files
            ]
            self._node_logger.info(f"Found {len(ply_urls)} PLY files for 3D viewer")

        # Send viewer data to frontend
        viewer_data = {
            "unique_id": unique_id,
            "default_tab": default_tab,
            "mp4_path": mp4_path,
            "mp4_b64": mp4_b64,
            "ply_dir": ply_dir,
            "ply_urls": ply_urls,
            "original_frames_dir": original_frames_dir,
            "autoplay": autoplay,
            "loop": loop,
            "title": iframe_title,
        }

        try:
            from server import PromptServer
            PromptServer.instance.send_sync("4k4d.viewer.load", viewer_data)
        except Exception as e:
            self._node_logger.warning(f"Failed to send viewer data to frontend: {e}")

        # Generate iframe HTML
        iframe_html_path = os.path.join(dataset_root, f"{name}_viewer.html")
        self._generate_iframe_html(iframe_html_path, mp4_path, iframe_title, autoplay, loop)

        embed_code = f'<iframe src="{iframe_html_path}" width="800" height="600" frameborder="0"></iframe>'

        return (iframe_html_path, embed_code)

    def _generate_iframe_html(self, output_path, mp4_path, title, autoplay, loop):
        """Generate a self-contained HTML file for embedding."""
        autoplay_attr = "autoplay" if autoplay else ""
        loop_attr = "loop" if loop else ""

        video_section = ""
        if mp4_path and os.path.exists(mp4_path):
            # Check file size for inline embedding
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            if size_mb < 50:
                try:
                    with open(mp4_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    video_section = f'''
                    <video {autoplay_attr} {loop_attr} controls style="width:100%;max-height:80vh;">
                        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                    </video>'''
                except Exception:
                    video_section = f'''
                    <video {autoplay_attr} {loop_attr} controls style="width:100%;max-height:80vh;">
                        <source src="file://{mp4_path}" type="video/mp4">
                    </video>'''
            else:
                video_section = f'''
                <p>Video file too large for inline embedding ({size_mb:.1f}MB).</p>
                <p>File path: {mp4_path}</p>'''

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #1a1a1a; color: #fff; font-family: sans-serif; }}
        h1 {{ font-size: 1.5em; margin-bottom: 10px; }}
        .viewer-container {{ max-width: 1200px; margin: 0 auto; }}
        video {{ border-radius: 8px; }}
        .info {{ color: #888; font-size: 0.85em; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="viewer-container">
        <h1>{title}</h1>
        {video_section}
        <p class="info">Generated by ComfyUI-4K4D</p>
    </div>
</body>
</html>'''

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html)

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

        # Send viewer data to frontend
        viewer_data = {
            "unique_id": unique_id,
            "default_tab": default_tab,
            "mp4_path": mp4_path,
            "mp4_b64": mp4_b64,
            "ply_dir": ply_dir,
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

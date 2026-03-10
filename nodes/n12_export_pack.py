"""
ComfyUI-4K4D Node 12: Export Pack
===================================
Packages all outputs for Premiere/Resolve import.
H.264, ProRes, PNG sequences, PLY clouds, and manifest JSON.
"""

import json
import logging
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

from ..core.base_node import BaseEasyVolcapNode
from ..core.constants import CATEGORIES, DATASET_INFO_TYPE

logger = logging.getLogger("4K4D.n12_export_pack")


class FourK4D_ExportPack(BaseEasyVolcapNode):
    """
    Packages all pipeline outputs into a download-ready archive.

    Creates a structured export directory with:
    - MP4 H.264 (web/preview)
    - ProRes 4444 MOV (Premiere/Resolve mastered quality)
    - PNG frame sequence (full quality NLE workflow)
    - PLY point clouds
    - Export manifest JSON
    - Step-by-step NLE import instructions
    """

    CATEGORY = CATEGORIES["output"]
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("export_manifest", "export_path", "total_size_gb", "premiere_import_guide")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_info": (DATASET_INFO_TYPE,),
            },
            "optional": {
                "export_path": ("STRING", {"default": ""}),
                "include_model": ("BOOLEAN", {"default": True}),
                "include_ply": ("BOOLEAN", {"default": True}),
                "include_h264_mp4": ("BOOLEAN", {"default": True}),
                "include_prores": ("BOOLEAN", {"default": True}),
                "include_png_sequence": ("BOOLEAN", {"default": False}),
                "include_configs": ("BOOLEAN", {"default": True}),
                "compress_output": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def execute(self, dataset_info, export_path="", include_model=True,
                include_ply=True, include_h264_mp4=True, include_prores=True,
                include_png_sequence=False, include_configs=True,
                compress_output=False, unique_id=None):
        return self._safe_execute(
            self._run, dataset_info, export_path, include_model, include_ply,
            include_h264_mp4, include_prores, include_png_sequence,
            include_configs, compress_output, unique_id
        )

    def _run(self, dataset_info, export_path, include_model, include_ply,
             include_h264_mp4, include_prores, include_png_sequence,
             include_configs, compress_output, unique_id):
        self._validate_dataset_info(dataset_info, ["dataset_root", "dataset_name"])

        dataset_root = dataset_info["dataset_root"]
        name = dataset_info["dataset_name"]

        if not export_path:
            export_path = os.path.join(self.env.paths["exports"], name)

        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "name": name,
            "created": datetime.now().isoformat(),
            "version": "2.0",
            "contents": {},
        }

        # Copy H.264 MP4
        if include_h264_mp4:
            render_output = dataset_info.get("render_output", "")
            if render_output:
                for mp4 in Path(render_output).rglob("*.mp4"):
                    dest = export_dir / "video" / mp4.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mp4, dest)
                    manifest["contents"]["h264_mp4"] = str(dest)

        # Copy ProRes
        if include_prores:
            render_output = dataset_info.get("render_output", "")
            if render_output:
                for mov in Path(render_output).rglob("*.mov"):
                    dest = export_dir / "prores" / mov.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mov, dest)
                    manifest["contents"]["prores_mov"] = str(dest)

        # Copy PNG sequence
        if include_png_sequence:
            render_output = dataset_info.get("render_output", "")
            if render_output:
                frames_src = Path(render_output) / "frames"
                if frames_src.exists():
                    frames_dest = export_dir / "png_sequence"
                    shutil.copytree(str(frames_src), str(frames_dest), dirs_exist_ok=True)
                    manifest["contents"]["png_sequence"] = str(frames_dest)

        # Copy PLY files
        if include_ply:
            ply_sources = [
                os.path.join(dataset_root, "vhulls"),
                os.path.join(dataset_root, "surfs"),
            ]
            render_output = dataset_info.get("render_output", "")
            if render_output:
                ply_sources.append(os.path.join(render_output, "ply"))

            for src in ply_sources:
                if os.path.exists(src):
                    for ply in Path(src).rglob("*.ply"):
                        dest = export_dir / "ply" / ply.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(ply, dest)
            manifest["contents"]["ply_clouds"] = str(export_dir / "ply")

        # Copy trained model
        if include_model:
            model_path = dataset_info.get("model_path") or dataset_info.get("supercharged_path")
            if model_path and os.path.exists(str(model_path)):
                model_dest = export_dir / "model"
                if os.path.isdir(str(model_path)):
                    shutil.copytree(str(model_path), str(model_dest), dirs_exist_ok=True)
                else:
                    model_dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(model_path), model_dest)
                manifest["contents"]["model"] = str(model_dest)

        # Copy configs
        if include_configs:
            config_path = dataset_info.get("config_path")
            if config_path and os.path.exists(str(config_path)):
                configs_dest = export_dir / "configs"
                configs_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(config_path), configs_dest)
                manifest["contents"]["configs"] = str(configs_dest)

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in export_dir.rglob("*") if f.is_file()
        )
        total_size_gb = total_size / (1024 ** 3)
        manifest["total_size_gb"] = round(total_size_gb, 3)

        # Write manifest
        manifest_path = export_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Compress if requested
        if compress_output:
            archive_path = f"{export_path}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(str(export_dir), arcname=name)
            manifest["archive"] = archive_path

        # NLE import guide
        guide = self._generate_import_guide(name, export_dir, manifest)

        return (
            json.dumps(manifest, indent=2),
            str(export_dir),
            total_size_gb,
            guide,
        )

    def _generate_import_guide(self, name, export_dir, manifest):
        """Generate step-by-step NLE import instructions."""
        lines = [
            "=" * 50,
            f"NLE IMPORT GUIDE: {name}",
            "=" * 50,
            "",
            "PREMIERE PRO:",
            "1. File > Import > Navigate to the export folder",
            "2. For ProRes: Import the .mov file directly",
            "3. For PNG sequence: Select first frame, check 'Image Sequence'",
            "4. Drag to timeline, set sequence settings to match",
            "",
            "DAVINCI RESOLVE:",
            "1. Media page > Import folder",
            "2. ProRes files will be recognized automatically",
            "3. For PNG sequence: Right-click > Import as Timeline",
            "",
            f"EXPORT LOCATION: {export_dir}",
            f"TOTAL SIZE: {manifest.get('total_size_gb', 0):.2f} GB",
            "",
            "CONTENTS:",
        ]
        for key, path in manifest.get("contents", {}).items():
            lines.append(f"  - {key}: {path}")

        return "\n".join(lines)

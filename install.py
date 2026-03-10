"""
ComfyUI-4K4D Post-Install Hook
================================
Called by ComfyUI Manager after cloning the repository.
Installs pip-safe dependencies only; CUDA extensions are handled by Node 2.
"""

import subprocess
import sys
import os

def install():
    """Install pip-safe requirements."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ])

if __name__ == "__main__":
    install()

"""The backend versions and install rules.

To move the backend, move the ``external/sd-scripts`` submodule and
``SD_SCRIPTS_COMMIT`` together, then recheck every route's plan. The tests
fail if the two differ.
"""

from __future__ import annotations

# The notebooks' setup cells clone this.
REPO_URL = "https://github.com/Linaqruf/kohya-trainer"

# Must equal the gitlink of external/sd-scripts.
SD_SCRIPTS_COMMIT = "cb70b96858477e716a2096292ab8aab433277f38"
SD_SCRIPTS_DIR = "external/sd-scripts"

# sd-scripts needs torch 2.6 or later and does not pin it. We keep Colab's
# torch, and install the strict build only when torch is missing, older than
# TORCH_MIN, or strict_torch=True. 2.8.0 is the newer of the two versions
# upstream CI tests. CUDA 12.8 also covers RTX 50 series GPUs.
TORCH_MIN = (2, 6)
TORCH_STRICT_PACKAGES = ("torch==2.8.0", "torchvision==0.23.0")
TORCH_STRICT_INDEX = "https://download.pytorch.org/whl/cu128"

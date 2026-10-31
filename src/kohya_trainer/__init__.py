"""kohya-trainer: Colab-first LoRA training over a pinned kohya-ss/sd-scripts backend."""

from .train import quick_train

__version__ = "0.1.0"
__all__ = ["__version__", "quick_train"]

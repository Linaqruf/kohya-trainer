# Kohya Trainer

Train LoRAs on Google Colab in one click. The notebooks run
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), pinned to an exact upstream
commit, so they keep working when upstream changes.

## What it is for

> kohya-trainer trains LoRAs on Colab the way LoRA trainers already work: sd-scripts names for every setting, the usual folder and caption habits, and the steps people take to get from images to a finished LoRA.
> - The 1-click notebook is one press: add your images and train. Every setting has a default. It is not a cut-down trainer: its cell takes any sd-scripts setting, so a power user can use it too.
> - The advanced notebook runs the same steps one cell at a time, so you can check and redo each stage.
> - Nothing goes in unless it helps you finish a LoRA in one sitting.

## Notebooks

| Model | GPU | 1-click | Advanced (per-stage) |
|---|---|---|---|
| Illustrious / NoobAI | L4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/illustrious_lora.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/illustrious_lora_advanced.ipynb) |
| Anima | L4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/anima_lora.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/anima_lora_advanced.ipynb) |
| MOSS-Image | L4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/moss_image_lora.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/moss_image_lora_advanced.ipynb) |

### Other models

| Model | GPU | 1-click | Advanced (per-stage) |
|---|---|---|---|
| Stable Diffusion 1.5 | L4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/sd15_lora.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/sd15_lora_advanced.ipynb) |
| SDXL | L4 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/sdxl_lora.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/notebooks/sdxl_lora_advanced.ipynb) |

The notebooks ask Colab for an L4 GPU, which needs a
[paid Colab plan](https://colab.research.google.com/signup). Each notebook says how it runs on
other GPUs.

**1-click**: pick a base model, point it at your images, and press run. You get the LoRA and
sample images, and with `save_every_n_epochs` at 1 (the default) a file for each epoch. Any
other sd-scripts setting goes in `config_overrides` in the cell's code.

**Advanced**: the same steps in separate cells (download, dataset, tagging, configure, train),
so you can check and redo each stage.

Both can upload each saved epoch to the Hugging Face Hub and log to Weights & Biases.

## Your dataset

- A folder, a .zip, a link to a .zip, or a Hugging Face dataset repo (`user/name`).
- Subfolders named like `10_mychar` train with that many repeats each.
- Each image can have a .txt caption file with the same name. If no image has one, the WD
  tagger tags them. On MOSS-Image the tagger is off by default, so set `trigger_word`.

With Google Drive mounted (the default), the LoRA files go to `MyDrive/lora/<project>/output`.

## Tokens

Add these to Colab Secrets when you need them: `HF_TOKEN` (read access for gated models,
write access for uploads), `WANDB_API_KEY` and `CIVITAI_TOKEN`. The notebooks never write a
token to disk or print it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) does the training.
- The 2026 rebuild was written with [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5).

## License

[Apache-2.0](LICENSE.md). The training backend, kohya-ss/sd-scripts, is also Apache-2.0.

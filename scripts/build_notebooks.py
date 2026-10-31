"""Build notebooks/ from the registry: <key>_lora.ipynb (1-click) and
<key>_lora_advanced.ipynb (one cell per stage) for every route.

The notebooks are build output: edit this file, run it and commit the result.
Cells only read the form and call the package. A test rejects a def or class.

What the notebooks are for (README and CONTRIBUTING have the same words):

  kohya-trainer trains LoRAs on Colab the way LoRA trainers already work:
  sd-scripts names for every setting, the usual folder and caption habits,
  and the steps people take to get from images to a finished LoRA.
  - The 1-click notebook is one press: add your images and train. Every
    setting has a default. It is not a cut-down trainer: its cell takes any
    sd-scripts setting, so a power user can use it too.
  - The advanced notebook runs the same steps one cell at a time, so you can
    check and redo each stage.
  - Nothing goes in unless it helps you finish a LoRA in one sitting.

Form rules:
- A field is a setting people change to finish a LoRA. Rarer ones go in
  Configure or the presets. Field names are sd-scripts names.
- For an optional number, 0 means off or auto.
- An empty string field puts its hint in a ``placeholder``. Never put a colon
  in one: Colab's form parser turns it into invalid JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kohya_trainer import config, models, pins, registry, tagging, train

GENERATED_NOTE = "*Built by `scripts/build_notebooks.py` in the repo. Your changes stay in your copy.*"

# The schedulers people pick. The dropdown also takes any name in train.SCHEDULERS.
SCHEDULERS = ["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"]
_LOCON_CONV = dict(arg.split("=") for arg in train.NETWORK_DEFAULTS["locon"]["network_args"])


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n"),
    }


def notebook(cells: list[dict], row: registry.ModelRoute) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": row.gpu},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


PAID_PLAN_URL = "https://colab.research.google.com/signup"

DICTS_COMMENT = (
    "# Other sd-scripts settings go in config_overrides, dataset settings in dataset_overrides. "
    "Both win over the form and the preset."
)


def _preset(row: registry.ModelRoute, preset: dict | None) -> dict:
    """The route's preset, loaded once per render and passed to each part."""
    return config.load_preset(row.preset) if preset is None else preset


def _trains_bf16(row: registry.ModelRoute, preset: dict | None = None) -> bool:
    return _preset(row, preset).get("mixed_precision") == "bf16"


def gpu_note(row: registry.ModelRoute, preset: dict | None = None) -> str:
    note = (
        f"This notebook asks Colab for an {row.gpu} GPU, which needs a [paid Colab plan]({PAID_PLAN_URL}). "
        "To use another GPU, open Runtime > Change runtime type."
    )
    if not row.t4_practical:
        return note + (
            f" {row.display_name} needs bf16 and is too large for a T4 at a usable speed, so pick an L4 or an A100."
        )
    if _trains_bf16(row, preset):
        # The form's defaults (batch 2 at 1024) run out of memory on a T4.
        return note + (
            " A T4 has no native bf16 and is much slower. On a T4, set `train_batch_size` to 1 and "
            "`resolution` to 768 or lower."
        )
    return note + " A T4 is slower."


def _vpred_note(row: registry.ModelRoute, preset: dict | None = None) -> str:
    """The v-pred hint above custom_base, on routes without the custom_base_is_vpred box."""
    if _is_flow(row, preset) or row.custom_vpred_option:
        return ""
    note = (
        "#@markdown For a pasted v-pred model, set `v_parameterization`, `zero_terminal_snr`, "
        "`scale_v_pred_loss_like_noise_pred` and `debiased_estimation_loss` to True in `config_overrides`, "
        'and `sample_sampler` to "euler".'
    )
    if any(base.v_prediction for base in row.base_models):
        note += " The v-pred model in the list sets them for you."
    return note + "\n"


def _vpred_field(row: registry.ModelRoute, advanced: bool) -> str:
    if not row.custom_vpred_option:
        return ""
    if advanced:
        help_text = (
            "#@markdown Tick this for a v-pred `custom_base`, such as a NoobAI v-pred merge. It sets "
            "`v_parameterization`, `zero_terminal_snr`, `scale_v_pred_loss_like_noise_pred` and "
            '`debiased_estimation_loss` to True, and `sample_sampler` to "euler". It does nothing without a '
            "custom base."
        )
        if any(base.v_prediction for base in row.base_models):
            help_text += " The v-pred model in the list sets them for you."
    else:
        help_text = "#@markdown Tick this if the pasted model is v-prediction. It does nothing without a custom base."
    return help_text + '\ncustom_base_is_vpred = False #@param {type:"boolean"}\n'


CONVROT_HELP = (
    "#@markdown ConvRot INT8: on uses about half the GPU memory, RAM and download. off uses bf16, the "
    "reference quality. auto picks on only when GPU memory or RAM is short."
)


def _convrot_field(row: registry.ModelRoute, advanced: bool) -> str:
    if not row.convrot_int8:
        return ""
    help_text = CONVROT_HELP
    if advanced:
        help_text += " The files download here, so run this cell again after a change."
    choices = json.dumps(list(models.CONVROT_CHOICES))
    return help_text + f'\nconvrot_int8 = "auto" #@param {choices}\n'


def _caches_te(row: registry.ModelRoute, preset: dict | None = None) -> bool:
    return bool(_preset(row, preset).get("cache_text_encoder_outputs"))


def _is_flow(row: registry.ModelRoute, preset: dict | None = None) -> bool:
    """Flow-matching routes get flow settings instead of SNR and noise settings."""
    return "timestep_sampling" in _preset(row, preset)


def tagging_intro(row: registry.ModelRoute) -> str:
    if row.default_auto_tag:
        return "If no image has a caption file, the WD tagger tags them."
    return "If no image has a caption file, each one trains on the trigger word alone, so set `trigger_word`."


# Form defaults for settings a preset can leave out. Every preset must set the
# other values _markers reads.
FORM_FALLBACKS = {"text_encoder_lr": "", "lr_scheduler_num_cycles": 3, "lr_warmup_steps": 0.0, "min_snr_gamma": 5}


def _markers(row: registry.ModelRoute, preset: dict) -> dict[str, str]:
    """The value of each __NAME__ marker. A default written into a template
    wins over the preset, because form_values do."""
    values = {**FORM_FALLBACKS, **preset}
    return {
        "__REPO_URL__": pins.REPO_URL,
        "__GPU_NOTE__": gpu_note(row, preset),
        "__ROW_KEY__": row.key,
        "__ROW_TITLE__": row.display_name,
        "__DEFAULT_BASE__": row.default_base,
        "__DEFAULT_RESOLUTION__": str(row.default_resolution),
        "__DEFAULT_BATCH__": str(row.default_batch_size),
        "__DEFAULT_AUTO_TAG__": str(row.default_auto_tag),
        "__TAGGING_INTRO__": tagging_intro(row),
        "__ABOUT_NOTE__": f"{row.about_note}\n\n" if row.about_note else "",
        "__DEFAULT_DIM__": str(values["network_dim"]),
        "__DEFAULT_ALPHA__": str(values["network_alpha"]),
        "__DEFAULT_LR__": str(values["learning_rate"]),
        "__DEFAULT_TE_LR__": str(values["text_encoder_lr"]),
        "__DEFAULT_EPOCHS__": str(values["max_train_epochs"]),
        "__DEFAULT_SCHEDULER__": str(values["lr_scheduler"]),
        "__DEFAULT_CYCLES__": str(values["lr_scheduler_num_cycles"]),
        "__DEFAULT_WARMUP__": str(values["lr_warmup_steps"]),
        "__DEFAULT_MIN_SNR__": str(values["min_snr_gamma"]),
        "__DEFAULT_SAVE_EVERY__": str(values["save_every_n_epochs"]),
        "__DEFAULT_SAMPLE_EVERY__": str(values["sample_every_n_epochs"]),
        "__DEFAULT_SEED__": str(values["seed"]),
        "__DEFAULT_CONV_DIM__": _LOCON_CONV["conv_dim"],
        "__DEFAULT_CONV_ALPHA__": _LOCON_CONV["conv_alpha"],
        "__REPEATS_TARGET__": str(row.repeats_target),
        "__NETWORK_TYPE_LIST__": json.dumps(list(row.network_types)),
        "__BASE_LIST__": json.dumps([b.key for b in row.base_models]),
        "__SCHEDULER_LIST__": json.dumps(SCHEDULERS),
        "__OPTIMIZER_LIST__": json.dumps(list(train.OPTIMIZERS)),
        "__DEFAULT_TAGGER__": tagging.DEFAULT_TAGGER,
        "__TAGGER_LIST__": json.dumps(list(tagging.TAGGERS)),
        "__DEFAULT_NEGATIVE__": row.default_sample_negative,
        "__ENTRYPOINT__": row.entrypoint,
        "__DICTS_COMMENT__": DICTS_COMMENT,
        "__OPTIMIZER_HELP__": OPTIMIZER_HELP,
        "__SCHEDULER_HELP__": SCHEDULER_HELP,
        "__PRUNE_HELP__": PRUNE_HELP,
        "__LEAK_NOTE__": LEAK_NOTE,
    }


def fill(
    template: str, row: registry.ModelRoute, preset: dict | None = None, sections: dict[str, str] | None = None
) -> str:
    """Replace the sections, then the __NAME__ markers. A section key that
    ends in a line break drops the whole line when its text is empty."""
    for marker, text in (sections or {}).items():
        template = template.replace(marker, text)
    for marker, value in _markers(row, _preset(row, preset)).items():
        template = template.replace(marker, value)
    return template


# --- form parts that depend on the route ----------------------------------

TE_LR_FIELD = 'text_encoder_lr = __DEFAULT_TE_LR__ #@param {type:"number"}\n'
TE_LR_FORM = '"text_encoder_lr": text_encoder_lr,\n'

SNR_FIELDS = """
#@markdown Min-SNR gamma: 5 is usual, 0 turns it off. Most current recipes leave `noise_offset` at 0. A v-pred base drops both.
min_snr_gamma = __DEFAULT_MIN_SNR__ #@param {type:"number"}
noise_offset = 0 #@param {type:"number"}
"""
SNR_FORM = '"min_snr_gamma": min_snr_gamma,\n"noise_offset": noise_offset,\n'

# The flow samplers the form offers, plus the preset's own if it is another.
FLOW_SAMPLERS = ["sigmoid", "shift"]
# Which samplers read each setting (get_noisy_model_input_and_timesteps and
# _SHIFT_AWARE_TIMESTEP_SAMPLING in library/flux_train_utils.py).
FLOW_SHIFT_SAMPLERS = ("sigma", "shift", "truncated_logit_normal")
FLOW_SCALE_SAMPLERS = ("sigmoid", "shift", "flux_shift")


def _quoted_list(names: list[str]) -> str:
    quoted = [f'"{name}"' for name in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def flow_samplers(row: registry.ModelRoute, preset: dict | None = None) -> list[str]:
    sampler = str(_preset(row, preset)["timestep_sampling"])
    return FLOW_SAMPLERS + ([sampler] if sampler not in FLOW_SAMPLERS else [])


def flow_fields(row: registry.ModelRoute, preset: dict | None = None) -> str:
    """The flow-matching fields: the preset's defaults, and a hint true for every sampler offered."""
    preset = _preset(row, preset)
    samplers = flow_samplers(row, preset)
    shift_users = [name for name in samplers if name in FLOW_SHIFT_SAMPLERS]
    scale_users = [name for name in samplers if name in FLOW_SCALE_SAMPLERS]
    help_text = f"`discrete_flow_shift` works only when `timestep_sampling` is {_quoted_list(shift_users)}."
    if preset["timestep_sampling"] == "sigmoid":
        help_text += " If hands and fine detail do not improve, try 3.0."
    elif preset["timestep_sampling"] == "truncated_logit_normal" and row.key == "moss_image":
        # MOSS-Image's figures: 1.0 is its training grid, and shift 3.0 comes close to its schedule.
        help_text += ' Keep 1.0 for "truncated_logit_normal". For "shift", use 3.0.'
    help_text += f" `sigmoid_scale` works with {_quoted_list(scale_users)}."
    if preset["timestep_sampling"] == "sigmoid" and preset["sigmoid_scale"] != 1.3:
        help_text += " Try 1.3 for sharper fine detail."
    return (
        f"#@markdown {help_text}\n"
        f'timestep_sampling = "{preset["timestep_sampling"]}" #@param {json.dumps(samplers)}\n'
        f'discrete_flow_shift = {preset["discrete_flow_shift"]} #@param {{type:"number"}}\n'
        f'sigmoid_scale = {preset["sigmoid_scale"]} #@param {{type:"number"}}\n'
    )


FLOW_FORM = (
    '"timestep_sampling": timestep_sampling,\n'
    '"discrete_flow_shift": discrete_flow_shift,\n'
    '"sigmoid_scale": sigmoid_scale,\n'
)

OPTIMIZER_HELP = (
    "#@markdown Type any sd-scripts optimizer, or one from an installed package as package.Name, such as "
    "pytorch_optimizer.CAME. CAME usually runs at 1e-4 to 2e-4. The optimizer defaults for Prodigy and "
    "ProdigyScheduleFree set your learning rate to 1.0."
)
SCHEDULER_HELP = (
    "#@markdown Pick one or type another sd-scripts name. Schedule-free optimizers ignore the scheduler "
    "and warmup. With Prodigy, cosine or constant_with_warmup are the usual picks."
)
LEAK_NOTE = (
    "- If every image shows one outfit, a prompt for that outfit can bring the subject in without the "
    "trigger word. If that is a problem, lower the LoRA weight."
)
PRUNE_HELP = (
    "After the tagger runs, this removes hair and eye tags found in most captions, so the trigger word learns the look."
)
MIN_SNR_FIELD = """
#@markdown Min-SNR gamma: 5 is usual, 0 turns it off.
min_snr_gamma = __DEFAULT_MIN_SNR__ #@param {type:"number"}
"""

BLOCK_SWAP_FIELD = """
#@markdown Blocks to keep in system RAM, to save GPU memory at a slower step. 0 turns it off. The most is __MAX_BLOCKS_TO_SWAP__.
blocks_to_swap = 0 #@param {type:"integer"}
"""


def block_swap_field(row: registry.ModelRoute) -> str:
    if not row.max_blocks_to_swap:
        return ""
    return BLOCK_SWAP_FIELD.strip("\n").replace("__MAX_BLOCKS_TO_SWAP__", str(row.max_blocks_to_swap)) + "\n"


CONV_FIELDS = """
#@markdown Only LoCon uses the conv layers.
conv_dim = __DEFAULT_CONV_DIM__ #@param {type:"integer"}
conv_alpha = __DEFAULT_CONV_ALPHA__ #@param {type:"number"}
"""
CONV_KWARGS = "conv_dim=conv_dim,\nconv_alpha=conv_alpha,\n"


# On the SDXL routes, full-size LoKr learns a character well at about this rate.
LOKR_STRONG_LR = 1e-3


def _short_rate(value: float) -> str:
    """A learning rate the way trainers write it: 3e-4, 2.5e-4, 1e-3."""
    if not value:
        return "0"
    mantissa, exponent = f"{value:.6e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _network_help(row: registry.ModelRoute, preset: dict | None = None) -> str:
    """The Network section header, for the network types this route offers."""
    preset = _preset(row, preset)
    dim, alpha = preset["network_dim"], preset["network_alpha"]
    if alpha * 2 == dim:
        pairing = "alpha to half of dim"
    elif alpha == dim:
        pairing = "alpha equal to dim"
    else:
        pairing = f"dim to {dim} and alpha to {alpha:g}"
    lines = [f"#@markdown **Network**. The defaults set {pairing}, so change them together."]
    if "loha" in row.network_types:
        lines.append(
            "#@markdown LoHa and LoKr ignore these two fields, run about twice as slow and need more GPU memory. "
            'To size them, add `"network_dim"` and `"network_alpha"` (LoHa, 16 and 8 by default) or '
            '`"network_args": ["factor=8"]` (LoKr) to `config_overrides`.'
            + (" On a T4, if it runs out of memory, set `train_batch_size` to 1." if row.t4_practical else "")
        )
    rate = preset["learning_rate"]
    if "lokr" in row.network_types and row.entrypoint == "sdxl_train_network.py" and rate < LOKR_STRONG_LR:
        lines.append(
            f"#@markdown If you pick lokr, set `learning_rate` to about {_short_rate(LOKR_STRONG_LR)}. At "
            f"{_short_rate(rate)}, full-size LoKr learns a character only weakly."
        )
    return "\n".join(lines) + "\n"


def _indent(fragment: str, spaces: int) -> str:
    pad = " " * spaces
    return "".join(f"{pad}{line}\n" for line in fragment.strip("\n").splitlines())


def route_fragments(row: registry.ModelRoute, preset: dict | None = None) -> dict[str, str]:
    """The template parts for one route. An empty string leaves that section out."""
    preset = _preset(row, preset)
    trains_te = not _caches_te(row, preset)
    has_locon = "locon" in row.network_types
    flow = _is_flow(row, preset)
    learning_fields = flow_fields(row, preset) if flow else SNR_FIELDS
    learning_form = FLOW_FORM if flow else SNR_FORM
    rates = row.entrypoint in tagging.RATING_TAG_ENTRYPOINTS
    return {
        "__NETWORK_HELP__": _network_help(row, preset),
        "__TE_LR_FIELD__": TE_LR_FIELD if trains_te else "",
        "__TE_LR_FORM_8__": _indent(TE_LR_FORM, 8) if trains_te else "",
        "__TE_LR_FORM_4__": _indent(TE_LR_FORM, 4) if trains_te else "",
        "__LEARNING_EXTRA_FIELDS__": learning_fields.strip("\n") + "\n",
        "__LEARNING_EXTRA_FORM_4__": _indent(learning_form, 4),
        "__CONV_FIELDS__": CONV_FIELDS.strip("\n") + "\n" if has_locon else "",
        "__CONV_KWARGS_4__": _indent(CONV_KWARGS, 4) if has_locon else "",
        "__RATING_FIELD__": RATING_FIELD.strip("\n") + "\n" if rates else "",
        "__RATING_KWARG_8__": _indent(RATING_KWARG, 8) if rates else "",
        "__BLOCK_SWAP_FIELD__": block_swap_field(row),
        "__BLOCK_SWAP_FORM_4__": _indent('"blocks_to_swap": blocks_to_swap,', 4) if row.max_blocks_to_swap else "",
        "__MIN_SNR_FIELD__": "" if flow else MIN_SNR_FIELD.strip("\n") + "\n",
        "__MIN_SNR_FORM_8__": "" if flow else _indent('"min_snr_gamma": min_snr_gamma,', 8),
        "__TRAINING_NOTE__": f"#@markdown {row.training_note}\n" if row.training_note else "",
        "__VPRED_KWARG_4__": _indent("custom_base_is_vpred=custom_base_is_vpred,", 4)
        if row.custom_vpred_option
        else "",
    }


def _route_sections(row: registry.ModelRoute, preset: dict) -> dict[str, str]:
    """route_fragments as fill sections, each marker with its line end."""
    return {marker + "\n": fragment for marker, fragment in route_fragments(row, preset).items()}


# --- shared cells ----------------------------------------------------------

SETUP_CELL = """
#@title Setup (installs the trainer in 2 to 4 minutes)
import os, subprocess, sys

ROOT = "/content/kohya-trainer"
if not os.path.isdir(ROOT):
    subprocess.run(["git", "clone", "--depth=1", "__REPO_URL__", ROOT], check=True)
    # Fetches the pinned commit, even after upstream moves on.
    subprocess.run(["git", "-C", ROOT, "submodule", "update", "--init", "--depth=1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ROOT], check=True)
# Python loads an editable install only at start, so add src for this session.
sys.path.insert(0, os.path.join(ROOT, "src"))

from kohya_trainer import environment
environment.install()
loaded = environment.load_tokens()
print("runtime:", environment.runtime_report())
print("tokens loaded from Colab Secrets:", ", ".join(loaded) or "none (add HF_TOKEN for gated or private Hugging Face repos, WANDB_API_KEY for log_to_wandb, CIVITAI_TOKEN for Civitai downloads that need a login)")
"""

# --- 1-click flavor ---------------------------------------------------------

ONECLICK_TITLE = (
    """
# __ROW_TITLE__ LoRA trainer (1-click)

Every setting has a default, so add your images and press run. __TAGGING_INTRO__
The LoRA files go to `MyDrive/lora/<project>/output` on Google Drive.

Any other sd-scripts setting goes in `config_overrides` in the cell's code. The
[advanced notebook](__REPO_URL__/blob/main/notebooks/__ROW_KEY___lora_advanced.ipynb)
runs the same steps one cell at a time, with the tagger settings and tag editing.

__GPU_NOTE__

__ABOUT_NOTE__"""
    + GENERATED_NOTE
)

ONECLICK_TRAIN = """
#@title Train
#@markdown **Project**
project_name = "my-lora" #@param {type:"string"}
mount_google_drive = True #@param {type:"boolean"}
#@markdown After a disconnect, tick this to continue from the last save. Keep the model and network fields the same.
resume_from_state = False #@param {type:"boolean"}
#@markdown ---
#@markdown **Model**
base_model = "__DEFAULT_BASE__" #@param __BASE_LIST__
custom_base = "" #@param {type:"string", placeholder:"a Hugging Face or Civitai link, used instead of the list"}
__VPRED_FIELD__
__CONVROT_FIELD__
#@markdown ---
#@markdown **Dataset**. A folder, a .zip, a link to a .zip, or a Hugging Face dataset repo (user/name). A subfolder named `10_mychar` trains with 10 repeats. A caption file is a .txt with the same name as its image. `auto_tag` tags the images only if none has one.
dataset = "/content/drive/MyDrive/dataset.zip" #@param {type:"string"}
auto_tag = __DEFAULT_AUTO_TAG__ #@param {type:"boolean"}
trigger_word = "" #@param {type:"string", placeholder:"a unique word such as mychar, added first to every caption"}
#@markdown __PRUNE_HELP__ It works only with a trigger word.
prune_appearance_tags = True #@param {type:"boolean"}
resolution = __DEFAULT_RESOLUTION__ #@param {type:"integer"}
#@markdown 0 picks the repeats for about __REPEATS_TARGET__ images per epoch.
num_repeats = 0 #@param {type:"integer"}
#@markdown ---
#@markdown **Training**
__TRAINING_NOTE__
max_train_epochs = __DEFAULT_EPOCHS__ #@param {type:"integer"}
#@markdown 0 uses the epoch count.
max_train_steps = 0 #@param {type:"integer"}
train_batch_size = __DEFAULT_BATCH__ #@param {type:"integer"}
__OPTIMIZER_HELP__
optimizer = "AdamW8bit" #@param __OPTIMIZER_LIST__ {allow-input: true}
learning_rate = __DEFAULT_LR__ #@param {type:"number"}
__TE_LR_FIELD__
__SCHEDULER_HELP__
lr_scheduler = "__DEFAULT_SCHEDULER__" #@param __SCHEDULER_LIST__ {allow-input: true}
#@markdown Warmup is a share of all steps (0.05 is 5%) or a step count. Only cosine_with_restarts uses the cycles.
lr_warmup_steps = __DEFAULT_WARMUP__ #@param {type:"number"}
lr_scheduler_num_cycles = __DEFAULT_CYCLES__ #@param {type:"integer"}
__MIN_SNR_FIELD__
#@markdown ---
__NETWORK_HELP__
network_type = "lora" #@param __NETWORK_TYPE_LIST__
network_dim = __DEFAULT_DIM__ #@param {type:"integer"}
network_alpha = __DEFAULT_ALPHA__ #@param {type:"number"}
#@markdown ---
#@markdown **Output**
#@markdown 0 saves only the final LoRA file.
save_every_n_epochs = __DEFAULT_SAVE_EVERY__ #@param {type:"integer"}
sample_prompt = "" #@param {type:"string", placeholder:"a prompt, or a .txt file with one per line (empty for the built-in one)"}
upload_to_hf = "" #@param {type:"string", placeholder:"username/repo for each saved epoch (needs a write HF_TOKEN)"}
#@markdown Needs the `WANDB_API_KEY` secret.
log_to_wandb = False #@param {type:"boolean"}

__DICTS_COMMENT__
config_overrides = {}
dataset_overrides = {}

from kohya_trainer import environment, train
train.check_project_name(project_name)
if mount_google_drive:
    environment.mount_drive()
plan = train.quick_train(
    model="__ROW_KEY__",
    base=custom_base or base_model,
    dataset=dataset,
    project_name=project_name,
    trigger_word=trigger_word,
    auto_tag=auto_tag,
    prune_appearance_tags=prune_appearance_tags,
    resume=resume_from_state,
    upload_repo=upload_to_hf,
    log_to_wandb=log_to_wandb,
__VPRED_KWARG_4__
__CONVROT_KWARG_4__
    network_type=network_type,
    network_dim=network_dim,
    network_alpha=network_alpha,
    resolution=resolution,
    train_batch_size=train_batch_size,
    num_repeats=num_repeats or None,
    sample_prompt=sample_prompt or None,
    optimizer=optimizer,
    form_values={
        "learning_rate": learning_rate,
__TE_LR_FORM_8__
        "max_train_epochs": max_train_epochs,
        "max_train_steps": max_train_steps,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
__MIN_SNR_FORM_8__
        "save_every_n_epochs": save_every_n_epochs,
    },
    config_overrides=config_overrides,
    dataset_overrides=dataset_overrides,
)
"""

ONECLICK_FOOTER = """
### After training

- The LoRA files are in `MyDrive/lora/<project_name>/output/`, and the sample images in
  `output/sample/`. Without Drive, they are in `/content/lora/<project_name>/output/`, which
  Colab deletes when the session ends.
- With Drive mounted and `save_every_n_epochs` at 1 (the default), every finished epoch is already
  saved, so a run that Colab stops keeps them.
- A session keeps running and using compute units after the page disconnects. If the page
  comes back on a new session, stop the old one in Runtime > Manage sessions. When you are
  done, choose Runtime > Disconnect and delete runtime.
__LEAK_NOTE__
__CAPTION_NOTE__
"""

# A preset that caches text-encoder outputs cannot shuffle captions.
ONECLICK_NOTE_TE_CACHED = """\
- Tags are not shuffled, because the text-encoder outputs are cached to save memory and time.
  The advanced notebook's notes say how to shuffle."""
ONECLICK_NOTE_TE_LIVE = """\
- Tags are shuffled, and `keep_tokens = 1` keeps your trigger word first."""

# --- advanced flavor --------------------------------------------------------

ADVANCED_TITLE = (
    """
# __ROW_TITLE__ LoRA trainer (advanced)

One cell per stage, so you can check and redo each one. The override dicts in
Configure take any sd-scripts setting.

__GPU_NOTE__

__ABOUT_NOTE__"""
    + GENERATED_NOTE
)

ADVANCED_PROJECT = """
#@title Project
project_name = "my-lora" #@param {type:"string"}
workspace_root = "" #@param {type:"string", placeholder:"empty uses Drive if mounted, else /content/lora"}
mount_google_drive = True #@param {type:"boolean"}
#@markdown After a disconnect, tick this to continue from the last save. Keep the model and network fields the same.
resume_from_state = False #@param {type:"boolean"}

from pathlib import Path
from kohya_trainer import environment, train
train.check_project_name(project_name)
if mount_google_drive:
    environment.mount_drive()
workspace = (Path(workspace_root) if workspace_root else environment.default_workspace()) / project_name
workspace.mkdir(parents=True, exist_ok=True)
print("workspace:", workspace)
"""

ADVANCED_BASE = """
#@title Base model
base_model = "__DEFAULT_BASE__" #@param __BASE_LIST__
__VPRED_NOTE__
custom_base = "" #@param {type:"string", placeholder:"a Hugging Face, Civitai or direct .safetensors link"}
__VPRED_FIELD__
__CONVROT_FIELD__

from kohya_trainer import models, registry
route = registry.get("__ROW_KEY__")
base_spec, checkpoint_path, vae_path, text_encoder_path = models.prepare_base(
    route, None if custom_base else base_model, custom=custom_base__CONVROT_ARG__)
print("checkpoint:", checkpoint_path)
print("vae:", vae_path)
print("text encoder:", text_encoder_path)
"""

ADVANCED_DATASET = """
#@title Dataset
#@markdown A folder, a .zip, a link to a .zip, or a Hugging Face dataset repo of image files or one .zip (user/name). A subfolder named `10_mychar` trains with 10 repeats. A caption file is a .txt with the same name as its image.
dataset_source = "/content/drive/MyDrive/dataset.zip" #@param {type:"string"}

from kohya_trainer import dataset, environment
dataset_dir = dataset.prepare_dataset(dataset_source, environment.scratch_dir(project_name))
print("dataset dir:", dataset_dir)
print(dataset.describe(dataset_dir))
"""

ADVANCED_TAG = """
#@title Captions (tagging and the trigger word)
#@markdown If ticked, the tagger replaces caption files after a copy to `captions-backup` in the workspace. To add to them, tick `append_tags`. To keep them, untick this.
run_tagger = __DEFAULT_AUTO_TAG__ #@param {type:"boolean"}
tagger = "__DEFAULT_TAGGER__" #@param __TAGGER_LIST__ {allow-input: true}
general_threshold = 0.35 #@param {type:"slider", min:0.1, max:0.9, step:0.05}
character_threshold = 0.35 #@param {type:"slider", min:0.1, max:0.9, step:0.05}
undesired_tags = "" #@param {type:"string", placeholder:"tags the tagger leaves out, such as watermark, signature"}
append_tags = False #@param {type:"boolean"}
character_tags_first = False #@param {type:"boolean"}
__RATING_FIELD__
#@markdown Removed from every caption file, with or without the tagger. `long_hair` matches `long hair`.
remove_tags = "" #@param {type:"string", placeholder:"such as simple background, white background"}
trigger_word = "" #@param {type:"string", placeholder:"a unique word such as mychar, added first to every caption (empty keeps your tags)"}
#@markdown __PRUNE_HELP__ It works only with a trigger word, and not with `append_tags`.
prune_appearance_tags = True #@param {type:"boolean"}

from kohya_trainer import dataset, tagging
# If you change this, set the same "caption_extension" in dataset_overrides in Configure.
caption_extension = ".txt"
if run_tagger:
    tagging.tag_images(
        dataset_dir,
        repo_id=tagger,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        undesired_tags=undesired_tags,
        append_tags=append_tags,
        character_tags_first=character_tags_first,
__RATING_KWARG_8__
        caption_extension=caption_extension,
        backup_dir=workspace / "captions-backup",
    )
    if prune_appearance_tags and trigger_word.strip() and not append_tags:
        tagging.prune_into_trigger(dataset_dir, caption_extension=caption_extension)
if remove_tags:
    tagging.remove_tags(dataset_dir, remove_tags, caption_extension=caption_extension)
# Put unwanted common tags in undesired_tags or remove_tags, then run this cell again.
print(dataset.tag_report(dataset_dir, caption_extension))
"""

RATING_FIELD = """
#@markdown Start each caption with its rating tag (general, sensitive, questionable or explicit).
use_rating_tags = False #@param {type:"boolean"}
"""
RATING_KWARG = "use_rating_tags=use_rating_tags,\n"

ADVANCED_CONFIGURE = """
#@title Configure
__NETWORK_HELP__
network_type = "lora" #@param __NETWORK_TYPE_LIST__
network_dim = __DEFAULT_DIM__ #@param {type:"integer"}
network_alpha = __DEFAULT_ALPHA__ #@param {type:"number"}
__CONV_FIELDS__
network_weights = "" #@param {type:"string", placeholder:"a LoRA file or link to keep training (empty starts fresh)"}
#@markdown ---
#@markdown **Training**
max_train_epochs = __DEFAULT_EPOCHS__ #@param {type:"integer"}
#@markdown 0 uses the epoch count.
max_train_steps = 0 #@param {type:"integer"}
train_batch_size = __DEFAULT_BATCH__ #@param {type:"integer"}
#@markdown Works like a bigger batch, with no extra GPU memory.
gradient_accumulation_steps = 1 #@param {type:"integer"}
__BLOCK_SWAP_FIELD__
resolution = __DEFAULT_RESOLUTION__ #@param {type:"integer"}
#@markdown 0 picks the repeats for about __REPEATS_TARGET__ images per epoch.
num_repeats = 0 #@param {type:"integer"}
seed = __DEFAULT_SEED__ #@param {type:"integer"}
#@markdown ---
#@markdown **Learning rate**
learning_rate = __DEFAULT_LR__ #@param {type:"number"}
__TE_LR_FIELD__
__SCHEDULER_HELP__
lr_scheduler = "__DEFAULT_SCHEDULER__" #@param __SCHEDULER_LIST__ {allow-input: true}
#@markdown Only cosine_with_restarts uses the cycles.
lr_scheduler_num_cycles = __DEFAULT_CYCLES__ #@param {type:"integer"}
#@markdown A share of all steps (0.05 is 5%) or a step count.
lr_warmup_steps = __DEFAULT_WARMUP__ #@param {type:"number"}
__LEARNING_EXTRA_FIELDS__
#@markdown ---
#@markdown **Optimizer**. To leave out the optimizer's recommended settings and keep your own learning rate, untick `apply_optimizer_defaults`.
__OPTIMIZER_HELP__
optimizer = "AdamW8bit" #@param __OPTIMIZER_LIST__ {allow-input: true}
apply_optimizer_defaults = True #@param {type:"boolean"}
optimizer_args = "" #@param {type:"string", placeholder:"key=value settings, such as weight_decay=0.05 (yours replace the defaults)"}
#@markdown ---
#@markdown **Saving and output**
#@markdown 0 saves only the final LoRA file.
save_every_n_epochs = __DEFAULT_SAVE_EVERY__ #@param {type:"integer"}
#@markdown Keep only the newest N epoch files. 0 keeps them all.
save_last_n_epochs = 0 #@param {type:"integer"}
sample_prompt = "" #@param {type:"string", placeholder:"a prompt, or a .txt file with one per line (empty for the built-in one)"}
sample_negative_prompt = "__DEFAULT_NEGATIVE__" #@param {type:"string"}
sample_every_n_epochs = __DEFAULT_SAMPLE_EVERY__ #@param {type:"integer"}
upload_to_hf = "" #@param {type:"string", placeholder:"username/repo for each saved epoch (needs a write HF_TOKEN)"}
make_repo_public = False #@param {type:"boolean"}
#@markdown Needs the `WANDB_API_KEY` secret.
log_to_wandb = False #@param {type:"boolean"}

# The form's training fields. 0 turns a number off.
form_values = {
    "learning_rate": learning_rate,
__TE_LR_FORM_4__
    "max_train_epochs": max_train_epochs,
    "max_train_steps": max_train_steps,
    "gradient_accumulation_steps": gradient_accumulation_steps,
__BLOCK_SWAP_FORM_4__
    "seed": seed,
    "save_every_n_epochs": save_every_n_epochs,
    "save_last_n_epochs": save_last_n_epochs,
    "lr_scheduler": lr_scheduler,
    "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
    "lr_warmup_steps": lr_warmup_steps,
    "sample_every_n_epochs": sample_every_n_epochs,
__LEARNING_EXTRA_FORM_4__
}
__DICTS_COMMENT__
config_overrides = {}
dataset_overrides = {}

from kohya_trainer import train
plan = train.build_plan(
    route,
    base_spec,
    checkpoint_path=checkpoint_path,
    vae_path=vae_path,
    text_encoder_path=text_encoder_path,
    dataset_dir=dataset_dir,
    workspace=workspace,
    project_name=project_name,
    trigger_word=trigger_word,
    network_type=network_type,
    network_dim=network_dim,
    network_alpha=network_alpha,
__CONV_KWARGS_4__
    network_weights=network_weights,
    optimizer=optimizer,
    optimizer_args=optimizer_args,
    optimizer_recommended=apply_optimizer_defaults,
    resume=resume_from_state,
    upload_repo=upload_to_hf,
    upload_public=make_repo_public,
    log_to_wandb=log_to_wandb,
__VPRED_KWARG_4__
    resolution=resolution,
    train_batch_size=train_batch_size,
    num_repeats=num_repeats or None,
    sample_prompt=sample_prompt or None,
    sample_negative_prompt=sample_negative_prompt,
    form_values=form_values,
    config_overrides=config_overrides,
    dataset_overrides=dataset_overrides,
)
print("training config:", plan.config_file)
print("dataset config:", plan.dataset_config_file)
print("launch:", " ".join(plan.command))
"""

ADVANCED_TRAIN = """
#@title Train
from kohya_trainer import train
train.execute(plan)
"""

ADVANCED_FOOTER = """
### Notes

- After a change, run that cell and the cells after it. Nothing downloads twice.
- With `save_every_n_epochs = 1` (the preset default), every finished epoch is already saved. Keep the
  workspace on Drive, the default when mounted. Colab deletes /content when a session ends.
- A session keeps running and using compute units after the page disconnects. If the page comes back on a
  new session, stop the old one in Runtime > Manage sessions. When you are done, choose Runtime >
  Disconnect and delete runtime.
__LEAK_NOTE__
__CAPTION_NOTE__
__TUNING_NOTE__
- `config_overrides` takes any sd-scripts argument (`python external/sd-scripts/__ENTRYPOINT__ --help`).
  `dataset_overrides` takes the arguments of `kohya_trainer.dataset.build_dataset_config`: shuffle_caption,
  keep_tokens, flip_aug, caption_extension, class_tokens, caption_prefix, resolution, batch_size,
  num_repeats and the bucket settings. A numbered folder's own number and name (such as 10_mychar) win over
  num_repeats and class_tokens there. Other dataset settings, such as caption_dropout_rate or color_aug,
  move to `config_overrides`. Any other key is refused.
__EXTRA_SETTINGS__
- The presets in `src/kohya_trainer/presets/` give the form its defaults and fill in every setting the
  form does not show. The form wins over the preset, and the two dicts win over both. To change a value
  the form shows, change the field.
"""

ADVANCED_NOTE_TE_CACHED = """\
- Caption shuffle is off, because the text-encoder outputs are cached next to the images. To shuffle, add
  `"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None` to `config_overrides`
  and set `dataset_overrides = {"shuffle_caption": True, "keep_tokens": 1}`. This uses more GPU memory.
- If a caption, the trigger word, `caption_prefix` or the text encoder changes, Train rebuilds that cache.
  Run one training at a time on a folder, because a second run replaces the first run's cache."""
ADVANCED_NOTE_TE_LIVE = """\
- Tags are shuffled (`shuffle_caption = True`), and `keep_tokens = 1` keeps the trigger word first. The
  text encoder trains too, so its outputs are not cached. For a fixed order, set
  `dataset_overrides = {"shuffle_caption": False}`."""


def render_oneclick(row: registry.ModelRoute) -> str:
    preset = config.load_preset(row.preset)
    sections = {
        "__VPRED_FIELD__\n": _vpred_field(row, advanced=False),
        "__CONVROT_FIELD__\n": _convrot_field(row, advanced=False),
        "__CONVROT_KWARG_4__\n": _indent("convrot_int8=convrot_int8,", 4) if row.convrot_int8 else "",
        **_route_sections(row, preset),
        "__CAPTION_NOTE__": ONECLICK_NOTE_TE_CACHED if _caches_te(row, preset) else ONECLICK_NOTE_TE_LIVE,
    }
    cells = [
        md(fill(ONECLICK_TITLE, row, preset, sections)),
        code(fill(SETUP_CELL, row, preset, sections)),
        code(fill(ONECLICK_TRAIN, row, preset, sections)),
        md(fill(ONECLICK_FOOTER, row, preset, sections)),
    ]
    return serialize(notebook(cells, row))


# Settings people often add, for the advanced footer: (keys, dict, text, shows
# on this route). A test checks each key against the backend.
EXTRA_SETTINGS = (
    (
        ("caption_dropout_rate",),
        "config_overrides",
        ('`"caption_dropout_rate": 0.1` trains one image in ten without its caption.'),
        lambda row, preset: True,
    ),
    (
        ("save_every_n_steps", "save_last_n_steps"),
        "config_overrides",
        (
            '`"save_every_n_steps": 500, "save_last_n_steps": 1500` also saves every 500 steps and keeps '
            "only the step files from the last 1500 steps."
        ),
        lambda row, preset: True,
    ),
    (
        ("min_timestep", "max_timestep"),
        "config_overrides",
        (
            '`"min_timestep": 0, "max_timestep": 700` trains only the lower noise levels (of 1000), for '
            "detail over composition."
        ),
        lambda row, preset: not _is_flow(row, preset),
    ),
    (
        ("network_dropout",),
        "config_overrides",
        '`"network_dropout": 0.1` can help against overfitting.',
        lambda row, preset: True,
    ),
    (
        ("flip_aug",),
        "dataset_overrides",
        '`{"flip_aug": True}` also trains on mirrored images. Leave it off for an asymmetric subject.',
        lambda row, preset: True,
    ),
    (
        ("clip_skip",),
        "config_overrides",
        '`"clip_skip": 1` uses the last text-encoder layer. The preset uses 2, as anime SD1.5 checkpoints do.',
        lambda row, preset: row.key == "sd15",
    ),
    (
        ("multires_noise_iterations", "multires_noise_discount"),
        "config_overrides",
        ('`"multires_noise_iterations": 6, "multires_noise_discount": 0.3` widens the range of contrast and detail.'),
        lambda row, preset: not _is_flow(row, preset),
    ),
    (
        ("lr_decay_steps",),
        "config_overrides",
        (
            '`"lr_decay_steps": 0.1` with `lr_scheduler = "warmup_stable_decay"` (typed in) holds the rate '
            "after warmup and lowers it over the last 10% of the steps."
        ),
        lambda row, preset: True,
    ),
)


def extra_settings(row: registry.ModelRoute, preset: dict | None = None) -> list[tuple[tuple[str, ...], str, str]]:
    """The EXTRA_SETTINGS entries this route shows, as (keys, dict, text)."""
    preset = _preset(row, preset)
    return [(keys, where, text) for keys, where, text, shows in EXTRA_SETTINGS if shows(row, preset)]


def _caption_dropout_text(row: registry.ModelRoute, text: str, preset: dict) -> str:
    """The caption_dropout_rate entry: the route's default, and whether the text cache keeps it."""
    default = dict(row.dataset_defaults).get("caption_dropout_rate")
    if default:
        text = f'`caption_dropout_rate` is {default:g} by default for this model. `"caption_dropout_rate": 0` turns it off.'
    if row.entrypoint in config.TE_CACHE_KEEPS_DROPOUT:
        return text + " It works with the cached text-encoder outputs."
    if _caches_te(row, preset):
        return text + " It needs text-encoder caching turned off, as for caption shuffle above."
    return text


def _extra_settings_note(row: registry.ModelRoute, preset: dict | None = None) -> str:
    preset = _preset(row, preset)
    lines = ["- Settings people often add in Configure's code:"]
    for keys, where, text in extra_settings(row, preset):
        if keys == ("caption_dropout_rate",):
            text = _caption_dropout_text(row, text, preset)
        lines.append(f"  - {where}: {text}")
    return "\n".join(lines)


def render_advanced(row: registry.ModelRoute) -> str:
    preset = config.load_preset(row.preset)
    sections = {
        "__VPRED_NOTE__\n": _vpred_note(row, preset),
        "__VPRED_FIELD__\n": _vpred_field(row, advanced=True),
        "__CONVROT_FIELD__\n": _convrot_field(row, advanced=True),
        "__CONVROT_ARG__": ", convrot_int8=convrot_int8" if row.convrot_int8 else "",
        **_route_sections(row, preset),
        "__CAPTION_NOTE__": ADVANCED_NOTE_TE_CACHED if _caches_te(row, preset) else ADVANCED_NOTE_TE_LIVE,
        "__EXTRA_SETTINGS__": _extra_settings_note(row, preset),
        "__TUNING_NOTE__\n": f"{row.tuning_note}\n" if row.tuning_note else "",
    }
    cells = [
        md(fill(ADVANCED_TITLE, row, preset, sections)),
        code(fill(SETUP_CELL, row, preset, sections)),
        code(fill(ADVANCED_PROJECT, row, preset, sections)),
        code(fill(ADVANCED_BASE, row, preset, sections)),
        code(fill(ADVANCED_DATASET, row, preset, sections)),
        code(fill(ADVANCED_TAG, row, preset, sections)),
        code(fill(ADVANCED_CONFIGURE, row, preset, sections)),
        code(fill(ADVANCED_TRAIN, row, preset, sections)),
        md(fill(ADVANCED_FOOTER, row, preset, sections)),
    ]
    return serialize(notebook(cells, row))


def serialize(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def outputs() -> dict[Path, str]:
    """Each notebook's path and its content. The sync test goes through this."""
    rendered: dict[Path, str] = {}
    for row in registry.ROUTES.values():
        rendered[ROOT / "notebooks" / f"{row.key}_lora.ipynb"] = render_oneclick(row)
        rendered[ROOT / "notebooks" / f"{row.key}_lora_advanced.ipynb"] = render_advanced(row)
    return rendered


def main() -> None:
    for path, content in outputs().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # The sync test compares bytes, so the line ends must not depend on the platform.
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

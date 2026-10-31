"""The build_plan steps that fill a run's settings, and the notes they print."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from . import environment
from .dataset_checks import DATASET_FALLBACK_ARGS, route_training_defaults
from .network import NETWORK_MODULES
from .optimizers import (
    _resolve_settings,
    _RunSettings,
    check_optimizer,
    check_scheduler,
    optimizer_arg_values,
    optimizer_that_runs,
)
from .registry import BaseModel, ModelRoute

# Form numbers where 0 means off: the key is dropped, so the backend default applies.
# save_every_n_epochs=0 must never reach the backend, which divides by it (train_network.py).
FORM_ZERO_MEANS_OFF = frozenset(
    {
        "max_train_steps",
        "gradient_accumulation_steps",
        "save_every_n_epochs",
        "save_last_n_epochs",
        "sample_every_n_epochs",
        "min_snr_gamma",
        "noise_offset",
        "blocks_to_swap",
    }
)


def form_settings(form_values: dict | None, preset: dict) -> dict:
    """The form's training values as config entries.

    - A 0 in a FORM_ZERO_MEANS_OFF field becomes None, which removes the key.
    - max_train_steps above 0 drops max_train_epochs.
    - learning_rate also sets the preset's unet_lr, which the backend prefers for the U-Net.
    """
    settings = dict(form_values or {})
    for key in FORM_ZERO_MEANS_OFF & settings.keys():
        if not settings[key]:
            settings[key] = None
    if settings.get("max_train_steps"):
        settings["max_train_epochs"] = None
    if "learning_rate" in settings and "unet_lr" in preset and "unet_lr" not in settings:
        settings["unet_lr"] = settings["learning_rate"]
    return settings


def check_wandb(log_to_wandb: bool, log_with: str) -> None:
    """Refuse W&B logging without a W&B login. Without one, the run stops after caching."""
    if (log_to_wandb or log_with in ("wandb", "all")) and not environment.wandb_login_available():
        way_out = "untick log_to_wandb" if log_to_wandb else "remove log_with from config_overrides"
        asked = "log_to_wandb is on" if log_to_wandb else f"log_with is {log_with!r}"
        raise ValueError(
            f"{asked}, but no W&B login was found. Add WANDB_API_KEY in Colab Secrets "
            f"(wandb.ai/authorize), or {way_out}."
        )


def _user_and_form(
    preset: dict, form_values: dict | None, config_overrides: dict | None, moved: dict
) -> tuple[dict, dict]:
    """The user's settings (config_overrides and the moved dataset keys) and the form's, kept apart."""
    user = {**moved, **(config_overrides or {})}
    form = form_settings(form_values, preset)
    if user.get("max_train_steps") and "max_train_epochs" not in user:
        # The backend lets max_train_epochs win over max_train_steps.
        form["max_train_epochs"] = None
    return user, form


# The Anima and MOSS-Image trainers refuse these with blocks_to_swap (assert_extra_args
# in anima_train_network.py and moss_image_train_network.py).
BLOCK_SWAP_CONFLICTS = ("cpu_offload_checkpointing", "unsloth_offload_checkpointing")


def check_block_swap(route: ModelRoute, settings: dict, user_set: dict) -> dict:
    """Return what blocks_to_swap adds to this run, and refuse what the trainer stops on.

    ``settings`` are the run's values, ``user_set`` the keys the form or the
    user gave. With gradient checkpointing, block_swap_h2d_only becomes a default.
    """
    blocks = settings.get("blocks_to_swap") or 0
    if not blocks or not route.max_blocks_to_swap:
        return {}
    if blocks > route.max_blocks_to_swap:
        raise ValueError(
            f"blocks_to_swap is {blocks}, but {route.display_name} can swap at most {route.max_blocks_to_swap} "
            "blocks: the trainer keeps at least two on the GPU. Lower blocks_to_swap."
        )
    for key in BLOCK_SWAP_CONFLICTS:
        if settings.get(key):
            raise ValueError(
                f"blocks_to_swap cannot be used with {key}: the trainer stops on the pair. Set blocks_to_swap "
                f"to 0, or remove {key} from config_overrides."
            )
    checkpointing = bool(settings.get("gradient_checkpointing"))
    if settings.get("block_swap_h2d_only") and not checkpointing:
        raise ValueError(
            "block_swap_h2d_only needs gradient_checkpointing, and the trainer stops without it. Add "
            '"gradient_checkpointing": True to config_overrides, or set "block_swap_h2d_only" to False.'
        )
    if "block_swap_h2d_only" in user_set or not checkpointing:
        return {}
    return {"block_swap_h2d_only": True}


def _repo_folder(project_name: str) -> str:
    """The Hub folder for a project's uploads, or "" (the repo root) if the Hub refuses the name."""
    parts = project_name.split("/")
    if any(part in ("", ".", "..", ".git", ".cache") for part in parts):
        return ""
    return project_name


def _run_value(run_overrides: dict, preset: dict, key: str, default=None):
    """A setting's value in this run. ``run_overrides`` wins over the preset, even with None."""
    return run_overrides.get(key, preset.get(key, default))


def _vpred_base(route: ModelRoute, base: BaseModel) -> BaseModel:
    """Train a pasted base as v-prediction. A model from the list keeps its own setting."""
    if any(listed.key == base.key for listed in route.base_models):
        print(
            "note: custom_base_is_vpred does nothing for a model from the list, which has its own "
            "setting. It is for a model pasted in custom_base."
        )
    elif not base.v_prediction:
        base = dataclasses.replace(base, v_prediction=True)
        print(
            "custom_base_is_vpred is ticked, so this base trains as v-prediction (v_parameterization, "
            "zero_terminal_snr and the rest, with euler samples)."
        )
    return base


def _merge_overrides(route: ModelRoute, network_type: str, form: dict, user_overrides: dict) -> dict:
    """The run's settings over the preset: the form's, the user's on top, then the route's defaults."""
    run_overrides = {**form, **user_overrides}
    for key, value in route_training_defaults(route).items():
        run_overrides.setdefault(key, value)
    if isinstance(run_overrides.get("lr_scheduler_args"), str):
        # The backend reads a string letter by letter, and fails after caching.
        run_overrides["lr_scheduler_args"] = run_overrides["lr_scheduler_args"].split()
    if network_type != "lora":
        run_overrides.setdefault("network_module", NETWORK_MODULES[network_type])
    return run_overrides


def _optimizer_settings(
    run_overrides: dict,
    preset: dict,
    user_overrides: dict,
    form: dict,
    *,
    optimizer: str,
    optimizer_args: str,
    optimizer_recommended: bool,
    sd_scripts_dir: str | Path,
) -> _RunSettings:
    """Set the optimizer that runs, its recommended settings and optimizer_args, and return them."""
    runs = optimizer_that_runs(optimizer, user_overrides, preset)
    if runs:
        check_optimizer(runs, sd_scripts_dir)  # a typo fails here, not after caching
    resolved = _resolve_settings(
        preset,
        form,
        user_overrides,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        optimizer_recommended=optimizer_recommended,
    )
    if resolved.optimizer:
        run_overrides["optimizer_type"] = resolved.optimizer_type
    # Only form values are replaced: the form's 3e-4 ruins an adaptive optimizer's run.
    for key in resolved.replaced:
        run_overrides[key] = resolved.recommended[key]
    for key in resolved.follows_lr:
        run_overrides[key] = user_overrides["learning_rate"]
    if resolved.optimizer_args is not None:  # None: config_overrides removes them all
        optimizer_arg_values(resolved.optimizer_args)  # fails here, not after caching
        if resolved.optimizer_args:
            run_overrides["optimizer_args"] = resolved.optimizer_args
        elif preset.get("optimizer_args"):
            run_overrides["optimizer_args"] = None  # another optimizer runs without the preset's
    _recommended_note(resolved.friendly, resolved.recommended, resolved.replaced, resolved.optimizer_args is not None)
    if resolved.follows_lr:
        print(
            f"learning_rate in config_overrides ({user_overrides['learning_rate']}) also sets "
            f"{' and '.join(resolved.follows_lr)}, which the backend uses in its place."
        )
    return resolved


def _recommended_note(friendly: str | None, recommended: dict, replaced: list[str], args_kept: bool) -> None:
    """Say what the recommended settings changed in this run, and only that."""
    used = []
    rates = [key for key in replaced if key in ("learning_rate", "unet_lr", "text_encoder_lr")]
    if rates:
        used.append(f"a learning rate of {recommended[rates[0]]:g}, which replaces the form's learning rate")
    if "max_grad_norm" in replaced:
        used.append("max_grad_norm 0, which turns off gradient clipping, as the optimizer's package advises")
    if recommended.get("optimizer_args") and args_kept:
        used.append(
            f"the settings in optimizer_args ({' '.join(recommended['optimizer_args'])}), which your own "
            "optimizer_args replace one by one"
        )
    if used:
        print(
            f"Using the recommended {friendly} settings:\n"
            + "".join(f"- {item}\n" for item in used)
            + "To keep your own values, put them in config_overrides, or untick apply_optimizer_defaults in "
            "the advanced notebook."
        )


def _scheduler_settings(run_overrides: dict, preset: dict, user_overrides: dict, resolved: _RunSettings) -> None:
    """Check the scheduler, and drop the settings the backend refuses with it."""
    scheduler = resolved.scheduler
    if resolved.scheduler_unused == "schedule-free":
        # The backend makes a dummy scheduler and checks no scheduler setting
        # (get_scheduler_fix in library/optimizer.py).
        print(
            f"{resolved.optimizer_type} is schedule-free: it sets its own schedule, so the backend does not use "
            "lr_scheduler or lr_warmup_steps."
        )
    elif resolved.scheduler_unused == "class":
        # torch.optim.lr_scheduler classes take no num_warmup_steps, and the backend
        # raises on a warmup above 0 (get_scheduler_fix). So a class gets a warmup only if the user set one.
        if "lr_warmup_steps" not in user_overrides:
            if _run_value(run_overrides, preset, "lr_warmup_steps"):
                print(
                    "note: lr_scheduler_type is set, so the form's warmup is dropped. A scheduler class gets no "
                    'warmup unless "lr_warmup_steps" is in config_overrides.'
                )
            run_overrides["lr_warmup_steps"] = None
    elif resolved.scheduler_unused == "adafactor":
        # The "adafactor:<lr>" schedule refuses warmup, and the form always sends one.
        run_overrides["lr_warmup_steps"] = None
        print(
            "AdaFactor is in relative-step mode. It sets its own learning rate, so the warmup "
            "is dropped and the scheduler choice is ignored. To set them yourself, add "
            "relative_step=False and warmup_init=False to optimizer_args."
        )
    else:
        check_scheduler(scheduler, {**preset, **run_overrides})
        # The backend refuses warmup with constant, after latent caching
        # (wrap_check_needless_num_warmup_steps in optimizer.py).
        if scheduler == "constant":
            if _run_value(run_overrides, preset, "lr_warmup_steps"):
                print(
                    "lr_scheduler is constant, which takes no warmup, so lr_warmup_steps is dropped. "
                    "For a warmup, pick constant_with_warmup."
                )
            run_overrides["lr_warmup_steps"] = None
        if scheduler == "warmup_stable_decay" and not _run_value(run_overrides, preset, "lr_decay_steps"):
            print(
                "note: warmup_stable_decay with no lr_decay_steps keeps the learning rate flat to the end. "
                'For a decay, add "lr_decay_steps" (0.1 is the last 10% of the steps) to config_overrides.'
            )
    if scheduler != "cosine_with_restarts" and "lr_scheduler_num_cycles" not in user_overrides:
        # Other schedulers that read it make the rate fall and rise again.
        run_overrides["lr_scheduler_num_cycles"] = None


def _text_encoder_rate(run_overrides: dict, resolved: _RunSettings) -> None:
    """Make a text-encoder learning rate of 0 train the U-Net only (``_RunSettings.zero_te_rate``)."""
    if resolved.zero_te_rate:
        run_overrides["network_train_unet_only"] = True
        run_overrides["text_encoder_lr"] = None
        print(
            "text_encoder_lr is 0, so only the U-Net trains. At a rate of 0 the text encoder still takes "
            "the full time and learns nothing."
        )


def _block_swap_settings(run_overrides: dict, route: ModelRoute, preset: dict) -> None:
    """Check blocks_to_swap, and add the settings it needs."""
    swap_defaults = check_block_swap(route, {**preset, **run_overrides}, run_overrides)
    if swap_defaults:
        run_overrides.update(swap_defaults)
        print(
            f"blocks_to_swap is {run_overrides['blocks_to_swap']}, so block_swap_h2d_only is on too: the "
            "swapped blocks are only copied to the GPU, never back, which is faster for LoRA training. To "
            'turn it off, add "block_swap_h2d_only": False to config_overrides.'
        )


# Below this system RAM in GB, an fp16 model loads straight to the GPU. SDXL in fp32 is about 14 GB.
LOW_RAM_GB = 15


def _low_ram_settings(run_overrides: dict, preset: dict) -> None:
    """Load an fp16 model straight to the GPU on a machine with little RAM.

    lowram alone puts an fp32 copy on the GPU (match_mixed_precision), so full_fp16
    is set too. Only the SD and SDXL loader needs this. bf16 routes already load to the GPU,
    and get no full_bf16, which changes training for no gain.
    """
    if environment.total_ram_gb() >= LOW_RAM_GB:
        return
    precision = run_overrides.get("mixed_precision") or preset.get("mixed_precision")
    if precision == "fp16":
        run_overrides.setdefault("full_fp16", True)
        run_overrides.setdefault("lowram", True)
        print("low system RAM, so the model is held in fp16 and loaded straight to the GPU (full_fp16 and lowram)")


def _upload_settings(run_overrides: dict, upload_repo: str, upload_public: bool, project_name: str) -> None:
    """Upload each saved file to the Hub while training goes on.

    The token never goes in the config. The backend's HfApi reads HF_TOKEN
    from the environment.
    """
    run_overrides.setdefault("huggingface_repo_id", upload_repo)
    run_overrides.setdefault("async_upload", True)
    # With no path, every upload fails with only a log line. "" is the repo root.
    if "huggingface_path_in_repo" in run_overrides and run_overrides["huggingface_path_in_repo"] is None:
        run_overrides["huggingface_path_in_repo"] = ""
    if "huggingface_path_in_repo" not in run_overrides:
        folder = _repo_folder(project_name)
        if folder:
            repo = run_overrides["huggingface_repo_id"]
            print(f"uploads: each saved LoRA goes to {repo}, in the folder {folder}/")
        else:
            print(f"note: {project_name!r} cannot be a folder name on the Hub, so files upload to the repo root")
        run_overrides["huggingface_path_in_repo"] = folder
    if upload_public:
        run_overrides.setdefault("huggingface_repo_visibility", "public")
    if not os.environ.get("HF_TOKEN"):
        print(
            "note: upload_repo is set, but no HF_TOKEN is loaded. Uploads need a token with write access "
            "in Colab Secrets. Without one, uploads fail and training goes on."
        )


def _wandb_settings(run_overrides: dict, preset: dict, log_to_wandb: bool, project_name: str) -> None:
    """Check the W&B login, and log to W&B when log_to_wandb is on."""
    check_wandb(log_to_wandb, str(run_overrides.get("log_with") or preset.get("log_with") or ""))
    if log_to_wandb:
        # wandb reads WANDB_API_KEY from the environment. Retries share one W&B project.
        run_overrides.setdefault("log_with", "wandb")
        run_overrides.setdefault("log_tracker_name", project_name)


def _convrot_notes(
    route: ModelRoute, user_overrides: dict, checkpoint_path: Path, text_encoder_path: Path | None
) -> None:
    """Say when a ConvRot flag set to False cannot apply, because the file is already int8."""
    int8_files = {
        Path(model.int8_filename).name
        for model in (*route.base_models, route.default_text_encoder)
        if model is not None and model.int8_filename
    }
    for flag, path in (("convrot_int8", checkpoint_path), ("convrot_int8_text_encoder", text_encoder_path)):
        if user_overrides.get(flag) is False and path is not None and Path(path).name in int8_files:
            print(
                f"note: {flag} is False in config_overrides, but {Path(path).name} is already a ConvRot INT8 "
                "file, and the trainer uses it as it is. For the bf16 file, set convrot_int8 to off in the cell "
                "where the models download, and run that cell again."
            )


def _text_encoder_file(run_overrides: dict, route: ModelRoute, text_encoder_path: Path | None) -> None:
    """Pass the text encoder as its own backend argument, on routes like Anima."""
    if text_encoder_path is None:
        raise ValueError(
            f"route {route.key!r} needs a text-encoder artifact ({route.text_encoder_arg}). "
            "models.prepare_base downloads it and returns its path."
        )
    run_overrides.setdefault(route.text_encoder_arg, str(text_encoder_path))


def _dataset_kwargs(
    route: ModelRoute, preset: dict, run_overrides: dict, dataset_overrides: dict, trigger_word: str, sizes: dict
) -> dict:
    """The build_dataset_config settings: ``sizes``, the route's defaults, dataset_overrides, then the trigger."""
    dataset_kwargs = dict(sizes)
    dataset_kwargs.update({key: value for key, value in route.dataset_defaults if key not in DATASET_FALLBACK_ARGS})
    dataset_kwargs.update(dataset_overrides)
    if trigger_word:
        # caption_prefix also covers images with no caption file. The trigger
        # in class_tokens too puts it in their captions twice.
        prefix = f"{trigger_word},"
        extra = (dataset_kwargs.get("caption_prefix") or "").strip()
        if not extra and run_overrides.get("caption_prefix"):
            # In the backend, the trigger's prefix replaces this one, so join them here.
            extra = str(run_overrides.pop("caption_prefix")).strip()
        if extra == trigger_word:
            pass  # the bare trigger: keep the comma after it
        elif extra.startswith((f"{trigger_word},", f"{trigger_word} ")):
            prefix = extra  # it already starts with the trigger
        elif extra:
            prefix = f"{prefix} {extra}"
            print(f"trigger word and caption_prefix joined: every caption starts with {prefix!r}")
        dataset_kwargs["caption_prefix"] = prefix
    if trigger_word and dataset_kwargs.get("shuffle_caption") and "keep_tokens" not in dataset_kwargs:
        dataset_kwargs["keep_tokens"] = 1
        print("captions are shuffled and there is a trigger word, so keep_tokens=1 keeps the trigger first")
    preset_caches_te = bool(
        preset.get("cache_text_encoder_outputs") or preset.get("cache_text_encoder_outputs_to_disk")
    )
    effective_caches_te = bool(
        _run_value(run_overrides, preset, "cache_text_encoder_outputs")
        or _run_value(run_overrides, preset, "cache_text_encoder_outputs_to_disk")
    )
    trains_te = not _run_value(run_overrides, preset, "network_train_unet_only")
    if preset_caches_te and not effective_caches_te and not dataset_kwargs.get("shuffle_caption") and not trains_te:
        print(
            "note: an override turns off text-encoder output caching, but this run does not shuffle captions, "
            "so it does not need that. Remove the override to save GPU memory."
        )
    return dataset_kwargs


# Above this training size in pixels, a bf16 run on a T4 is very slow.
T4_BF16_MAX_SIZE = 768


def _t4_notes(
    route: ModelRoute, preset: dict, run_overrides: dict, network_type: str, size: tuple[int, int], batch: int
) -> None:
    """Warn, before latent caching, about settings that run out of memory or crawl on a T4."""
    on_t4 = "T4" in (environment.gpu_name() or "")
    if on_t4 and batch > 1 and network_type in ("loha", "lokr"):
        print(
            f"{network_type} needs more GPU memory than plain LoRA. If training runs out of memory "
            f"on this T4, set train_batch_size=1 (now {batch})."
        )
    trains_bf16 = (run_overrides.get("mixed_precision") or preset.get("mixed_precision")) == "bf16"
    # models.prepare_base warns about a route a T4 cannot train at all.
    if trains_bf16 and on_t4 and route.t4_practical and (max(size) > T4_BF16_MAX_SIZE or batch > 1):
        print(
            "note: this model trains in bf16, which a T4 does not support natively. On a T4, use "
            "train_batch_size = 1 (batch 2 at 1024 runs out of memory), and a resolution of 512 or 768 "
            "to keep it from being very slow."
        )


def _dropped_setting_notes(run_overrides: dict, train_config: dict, form: dict) -> None:
    """Say which settings the run cannot use, so no typed value is dropped without a word."""
    for dropped, cause in (("min_snr_gamma", "debiased_estimation_loss"), ("noise_offset", "zero_terminal_snr")):
        if run_overrides.get(dropped) and dropped not in train_config:
            print(f"{dropped} dropped: {cause} replaces it in this run (v-pred bases turn that flag on)")
    # The backend saves a state only next to an epoch or step file.
    if (
        "save_every_n_epochs" in form
        and not train_config.get("save_every_n_epochs")
        and not train_config.get("save_every_n_steps")
    ):
        print(
            "save_every_n_epochs is 0, so only the final LoRA is saved. A session that ends early "
            "leaves no file and no state to resume from."
        )

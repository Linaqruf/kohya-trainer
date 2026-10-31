"""build_plan, execute and quick_train, the one-click path.

The checks that protect a run live in build_plan, so the advanced notebook,
which calls it directly, gets them too."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config as config_mod
from . import dataset as dataset_mod
from . import environment, models, registry, tagging

# The _ names in these imports are package-internal helpers shared between the plan modules.
from .dataset_checks import (
    _caption_extension,
    _check_dataset,
    _extra_caption_text,
    _refuse_no_captions,
    _refuse_no_images,
    _step_estimate,
    caption_sources,
    check_caption_candidates,
    check_dataset_args,
    dataset_settings,
    route_training_defaults,
    split_dataset_overrides,
)
from .network import (
    _adapter_rate,
    _check_network_type,
    _network_args,
    _network_size,
    _network_weights,
    check_adapter_args,
    network_module_that_runs,
)
from .optimizers import _resolve_settings, check_optimizer, check_scheduler, optimizer_arg_values, optimizer_that_runs
from .previews import _image_size, _write_prompts, sample_prompt_lines
from .registry import BaseModel, ModelRoute
from .resume import _check_epochs_left, _resume_guard, _resume_settings
from .settings import (
    _block_swap_settings,
    _convrot_notes,
    _dataset_kwargs,
    _dropped_setting_notes,
    _low_ram_settings,
    _merge_overrides,
    _optimizer_settings,
    _scheduler_settings,
    _t4_notes,
    _text_encoder_file,
    _text_encoder_rate,
    _upload_settings,
    _user_and_form,
    _vpred_base,
    _wandb_settings,
    check_block_swap,
    check_wandb,
)


def check_project_name(project_name: str) -> None:
    """Refuse a project name that cannot be the workspace folder and the start of every file name."""
    name = project_name if isinstance(project_name, str) else ""
    if not name.strip() or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"project_name {project_name!r} cannot be used as a folder name. Use a short name without "
            '"/" or "\\", for example my-lora.'
        )


def check_before_downloads(route: ModelRoute, plan_kwargs: dict, dataset_dir: Path | None = None) -> None:
    """Run the checks of build_plan that need no model, so a typo fails before any download.

    With ``dataset_dir``, a folder already on disk, its captions are checked too.
    """
    check_dataset_args(plan_kwargs.get("config_overrides"))
    kept, moved, _ = split_dataset_overrides(plan_kwargs.get("dataset_overrides"))
    preset = config_mod.load_preset(route.preset)
    user, form = _user_and_form(preset, plan_kwargs.get("form_values"), plan_kwargs.get("config_overrides"), moved)
    network_type = plan_kwargs.get("network_type") or "lora"
    check_adapter_args(route, network_module_that_runs(network_type, user, preset), user.get("network_args"))
    sd_scripts_dir = plan_kwargs.get("sd_scripts_dir") or environment.sd_scripts_dir()
    runs = optimizer_that_runs(plan_kwargs.get("optimizer", ""), user, preset)
    if runs:
        check_optimizer(runs, sd_scripts_dir)
    resolved = _resolve_settings(
        preset,
        form,
        user,
        optimizer=plan_kwargs.get("optimizer", ""),
        optimizer_args=plan_kwargs.get("optimizer_args") or "",
        optimizer_recommended=plan_kwargs.get("optimizer_recommended", True),
    )
    settings = {**preset, **route_training_defaults(route), **form, **user}
    check_block_swap(route, settings, {**form, **user})
    if resolved.zero_te_rate:
        settings["network_train_unet_only"] = True  # build_plan turns a 0 rate into this
    config_mod.check_text_encoder_training({key: value for key, value in settings.items() if value is not None})
    if dataset_dir is not None and Path(dataset_dir).is_dir():
        extension = _caption_extension(kept, user)
        folders = dataset_mod.build_dataset_config(dataset_dir, caption_extension=extension)
        check_caption_candidates(
            {key: value for key, value in settings.items() if value is not None}, folders, route.entrypoint
        )
    optimizer_arg_values(resolved.optimizer_args)  # an entry the backend cannot read
    if not resolved.scheduler_unused:
        check_scheduler(resolved.scheduler, settings)
    sample_prompt_lines(plan_kwargs.get("sample_prompt"), "")
    check_wandb(bool(plan_kwargs.get("log_to_wandb")), str(settings.get("log_with") or ""))


@dataclass
class LaunchPlan:
    """One planned training run: the command, the files it reads and the step counts."""

    route_key: str
    base_key: str
    command: list[str]
    config_file: Path
    dataset_config_file: Path
    sample_prompts_file: Path
    output_dir: Path
    estimated_steps_per_epoch: int
    estimated_total_steps: int | None
    train_config: dict = field(repr=False)
    dataset_config: dict = field(repr=False)
    # Written to resume_guard_file when training starts, for the next resume's check.
    resume_guard: dict = field(default_factory=dict, repr=False)
    resume_guard_file: Path | None = None


def build_launch_command(
    entrypoint: str,
    config_file: str | Path,
    dataset_config: str | Path,
    *,
    sd_scripts_dir: str | Path,
    num_cpu_threads: int = 1,
) -> list[str]:
    """The accelerate command that runs ``entrypoint`` from the backend with the two config files."""
    return [
        "accelerate",
        "launch",
        f"--num_cpu_threads_per_process={num_cpu_threads}",
        str(Path(sd_scripts_dir) / entrypoint),
        f"--config_file={config_file}",
        f"--dataset_config={dataset_config}",
    ]


def _dataset_config(
    route: ModelRoute,
    preset: dict,
    run_overrides: dict,
    dataset_overrides: dict,
    dataset_dir: Path,
    trigger_word: str,
    *,
    resolution: int | None,
    train_batch_size: int | None,
    num_repeats: int | None,
) -> tuple[dict, list[dict], tuple[int, int], int]:
    """Check the dataset and return its config, subset counts, and the size and batch that train."""
    dataset_overrides = dataset_settings(run_overrides, dataset_overrides)
    subsets, image_count = _check_dataset(dataset_dir, run_overrides, dataset_overrides, trigger_word, num_repeats)
    if num_repeats is None:
        # Aimed at the images whose repeats no folder name sets.
        unnumbered = sum(s["images"] for s in subsets if not s["num_repeats"])
        num_repeats = dataset_mod.default_num_repeats(unnumbered or image_count, route.repeats_target)
    if resolution is None:
        resolution = route.default_resolution
    if train_batch_size is None:
        train_batch_size = route.default_batch_size
    sizes = {"resolution": resolution, "batch_size": train_batch_size, "num_repeats": num_repeats}
    dataset_kwargs = _dataset_kwargs(route, preset, run_overrides, dataset_overrides, trigger_word, sizes)
    # With a trigger word, folder names stay out of the captions, or the subject is named twice.
    dataset_config = dataset_mod.build_dataset_config(
        dataset_dir, **{"folder_class_tokens": not trigger_word, **dataset_kwargs}
    )
    size = _image_size(dataset_config["general"].get("resolution"), resolution)
    batch = dataset_config["datasets"][0]["batch_size"]
    return dataset_config, subsets, size, batch


def build_plan(
    route: ModelRoute,
    base: BaseModel,
    *,
    checkpoint_path: Path,
    vae_path: Path | None,
    dataset_dir: Path,
    workspace: Path,
    project_name: str,
    text_encoder_path: Path | None = None,
    trigger_word: str = "",
    resolution: int | None = None,
    train_batch_size: int | None = None,
    num_repeats: int | None = None,
    sample_prompt: str | None = None,
    sample_negative_prompt: str | None = None,
    network_type: str = "lora",
    network_dim: int | None = None,
    network_alpha: float | None = None,
    conv_dim: int | None = None,
    conv_alpha: float | None = None,
    network_weights: str = "",
    optimizer: str = "",
    optimizer_args: str = "",
    optimizer_recommended: bool = True,
    resume: bool | str | Path = False,
    upload_repo: str = "",
    upload_public: bool = False,
    log_to_wandb: bool = False,
    custom_base_is_vpred: bool = False,
    form_values: dict | None = None,
    config_overrides: dict | None = None,
    dataset_overrides: dict | None = None,
    sd_scripts_dir: Path | None = None,
) -> LaunchPlan:
    """Build the configs and command for one training run, and write them to disk.

    Where each training setting comes from, lowest first:

    1. The route's preset.
    2. ``form_values``, keyed by sd-scripts names. 0 means off or auto (``form_settings``).
    3. The recommended settings of the optimizer that runs. They replace only form values.
    4. ``config_overrides``. It wins over all, and a None in it removes the key.

    ``dataset_overrides`` wins over the form.

    - ``custom_base_is_vpred``: train a pasted base as v-prediction.
    - ``resume``: True takes the newest saved state, or starts fresh if there
      is none. A path resumes from that state folder.
    """
    check_project_name(project_name)
    check_dataset_args(config_overrides)
    dataset_overrides, moved_dataset_args, dataset_notes = split_dataset_overrides(dataset_overrides)
    for note in dataset_notes:
        print(f"note: {note}")
    sd_scripts_dir = sd_scripts_dir or environment.sd_scripts_dir()
    config_dir = workspace / "config"
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_network_type(route, network_type)
    trigger_word = (trigger_word or "").strip()
    preset = config_mod.load_preset(route.preset)
    if custom_base_is_vpred:
        base = _vpred_base(route, base)

    # The user's settings over the form's. The steps below add to it.
    user_overrides, form = _user_and_form(preset, form_values, config_overrides, moved_dataset_args)
    run_overrides = _merge_overrides(route, network_type, form, user_overrides)
    network_args = _network_args(run_overrides, network_type, conv_dim, conv_alpha)
    _network_size(run_overrides, network_type, network_dim, network_alpha)
    _adapter_rate(run_overrides, route, preset, network_type, network_args)
    if network_weights:
        _network_weights(run_overrides, route, network_type, network_args, network_weights, resume)
    resolved = _optimizer_settings(
        run_overrides,
        preset,
        user_overrides,
        form,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        optimizer_recommended=optimizer_recommended,
        sd_scripts_dir=sd_scripts_dir,
    )
    _scheduler_settings(run_overrides, preset, user_overrides, resolved)
    _text_encoder_rate(run_overrides, resolved)  # after the optimizer settings
    _block_swap_settings(run_overrides, route, preset)
    _low_ram_settings(run_overrides, preset)
    if upload_repo:
        _upload_settings(run_overrides, upload_repo, upload_public, project_name)
    _wandb_settings(run_overrides, preset, log_to_wandb, project_name)
    _convrot_notes(route, user_overrides, checkpoint_path, text_encoder_path)
    if route.text_encoder_arg:
        _text_encoder_file(run_overrides, route, text_encoder_path)
    continue_epoch, continue_note = None, ""
    if resume:
        continue_epoch, continue_note = _resume_settings(run_overrides, resume, output_dir, project_name)

    # dataset_overrides can change the size and batch the notes below use.
    dataset_config, subsets, size, batch = _dataset_config(
        route,
        preset,
        run_overrides,
        dataset_overrides,
        dataset_dir,
        trigger_word,
        resolution=resolution,
        train_batch_size=train_batch_size,
        num_repeats=num_repeats,
    )
    _t4_notes(route, preset, run_overrides, network_type, size, batch)

    train_config = config_mod.build_train_config(
        route,
        base,
        checkpoint_path=checkpoint_path,
        vae_path=vae_path,
        output_dir=output_dir,
        output_name=project_name,
        sample_prompts=config_dir / "sample_prompts.txt",
        overrides=run_overrides,
        sd_scripts_dir=sd_scripts_dir,
    )
    config_mod.check_compat(train_config, dataset_config, route.entrypoint)
    check_caption_candidates(train_config, dataset_config, route.entrypoint)
    prompts_file = _write_prompts(
        route,
        train_config,
        dataset_config,
        config_dir / "sample_prompts.txt",
        trigger_word,
        sample_prompt,
        sample_negative_prompt,
        size,
    )
    _check_epochs_left(continue_epoch, train_config)
    _dropped_setting_notes(run_overrides, train_config, form)
    guard_file = config_dir / "resume_guard.json"
    guard = _resume_guard(train_config, network_type, base, guard_file)
    if continue_epoch:
        # Printed only after every check above has passed.
        print(f"continuing with epoch {continue_epoch}{continue_note}")
    steps_per_epoch, total_steps = _step_estimate(train_config, dataset_config, subsets, batch)

    config_file = config_mod.write_toml(train_config, config_dir / "training.toml")
    dataset_config_file = config_mod.write_toml(dataset_config, config_dir / "dataset.toml")
    command = build_launch_command(route.entrypoint, config_file, dataset_config_file, sd_scripts_dir=sd_scripts_dir)
    return LaunchPlan(
        route_key=route.key,
        base_key=base.key,
        command=command,
        config_file=config_file,
        dataset_config_file=dataset_config_file,
        sample_prompts_file=prompts_file,
        output_dir=Path(train_config["output_dir"]),
        estimated_steps_per_epoch=steps_per_epoch,
        estimated_total_steps=total_steps,
        train_config=train_config,
        dataset_config=dataset_config,
        resume_guard=guard,
        resume_guard_file=guard_file,
    )


def record_resume_guard(plan: LaunchPlan) -> None:
    """Write what this run trains with, for the next resume's check.

    Only execute calls it, so a plan built only to look at it keeps the old record.
    """
    if plan.resume_guard_file is None:
        return
    plan.resume_guard_file.parent.mkdir(parents=True, exist_ok=True)
    plan.resume_guard_file.write_text(json.dumps(plan.resume_guard, indent=1) + "\n", encoding="utf-8", newline="\n")


def execute(plan: LaunchPlan, sd_scripts_dir: Path | None = None) -> None:
    """Run a plan and stream the backend's output. On failure, the error holds the end of the output.

    It returns nothing, because a notebook cell prints a returned value.
    """
    sd_scripts_dir = sd_scripts_dir or environment.sd_scripts_dir()
    if plan.train_config.get("log_with") in ("wandb", "all"):
        environment.ensure_wandb()  # not in the pinned backend requirements
    if plan.train_config.get("cache_text_encoder_outputs_to_disk"):
        # Here, not in build_plan, so planning never deletes files in the dataset.
        entrypoint = registry.get(plan.route_key).entrypoint
        dataset_mod.refresh_text_caches(plan.dataset_config, plan.train_config, entrypoint)
    record_resume_guard(plan)
    returncode, detail = environment.run_streamed(plan.command, cwd=sd_scripts_dir)
    if returncode != 0:
        raise RuntimeError(f"training exited with code {returncode}. Backend output tail:\n{detail}")
    print(f"Done. LoRA files are in {plan.output_dir}")


def _check_quick_inputs(
    route: ModelRoute, dataset: str | Path, base: str | None, convrot_int8: str, plan_kwargs: dict
) -> str:
    """Run the checks quick_train makes before any download or dataset work.

    Returns the pasted base, resolved once, or "" for a base from the list.
    """
    # A zip is checked later, in _ready_dataset, still before the models download.
    local = Path(str(dataset)).expanduser()
    check_before_downloads(route, plan_kwargs, dataset_dir=local if local.is_dir() else None)
    # The base and ConvRot checks fail before the dataset is unpacked and tagged.
    # Resolved once, so a Civitai link is looked up and warned about once.
    custom = models.custom_base(base, route.key) if base and "/" in base else ""
    if not custom:
        route.base(base)
    models.convrot_choice(route, convrot_int8, plan_kwargs.get("config_overrides"))
    return custom


def _ready_dataset(
    dataset: str | Path,
    project_name: str,
    trigger_word: str,
    auto_tag: bool,
    prune_appearance_tags: bool,
    plan_kwargs: dict,
) -> Path:
    """Unpack and tag the dataset, refuse one with nothing to train on, and return its folder."""
    # Local disk, never Drive: training reads the latent caches next to the images every epoch.
    dataset_dir = dataset_mod.prepare_dataset(dataset, environment.scratch_dir(project_name))

    caption_extension = _caption_extension(plan_kwargs.get("dataset_overrides"), plan_kwargs.get("config_overrides"))
    summary = dataset_mod.summarize(dataset_dir, caption_extension)
    subsets = dataset_mod.subset_counts(dataset_dir, caption_extension)
    print(
        f"dataset: {summary['images']} images, {summary['captioned']} captioned"
        + (f", in {len(subsets)} folders" if len(subsets) > 1 else "")
    )
    _refuse_no_images(dataset_dir, summary, "`dataset`")
    # The tagger overwrites caption files, so it runs only when no subset has any.
    if auto_tag and summary["captioned"] == 0:
        print("no caption files found, so the WD tagger runs first")
        tagging.tag_images(dataset_dir, caption_extension=caption_extension)
        if trigger_word and prune_appearance_tags:
            tagging.prune_into_trigger(dataset_dir, caption_extension)
        summary = dataset_mod.summarize(dataset_dir, caption_extension)
        subsets = dataset_mod.subset_counts(dataset_dir, caption_extension)
    # Without a trigger, uncaptioned images in a 10_name folder train on the name.
    dataset_extra = plan_kwargs.get("dataset_overrides") or {}
    extra_text = _extra_caption_text(dataset_extra, plan_kwargs.get("config_overrides") or {})
    sources = caption_sources(subsets, trigger_word, dataset_extra.get("class_tokens"), extra_text)
    _refuse_no_captions(sources, "Turn on auto_tag, add .txt caption files, or set a trigger_word.")
    if auto_tag and sources["empty"] and sources["captioned"]:
        # build_plan says these images train on an empty caption. Only here can we say why.
        print(
            "auto_tag did not run, because some images have caption files. To tag every "
            "image, remove all the caption files and run this cell again."
        )
    return dataset_dir


def quick_train(
    *,
    model: str,
    dataset: str | Path,
    project_name: str,
    base: str | None = None,
    trigger_word: str = "",
    auto_tag: bool | None = None,
    prune_appearance_tags: bool = True,
    workspace: str | Path | None = None,
    dry_run: bool = False,
    convrot_int8: str = "auto",
    **plan_kwargs,
) -> LaunchPlan:
    """The one-click path: download, tag, plan and train. Returns the plan.

    - ``base``: a key from the route's list, or ``user/repo`` or ``user/repo:file.safetensors``.
    - ``dry_run``: build and write everything, but do not train.
    - ``convrot_int8``: "auto", "on" or "off", on routes with ConvRot INT8 files.
    - ``auto_tag``: tag the images when none has a caption file. None takes the route's default.
    - ``prune_appearance_tags``: with a trigger word, remove the hair and eye
      tags the tagger wrote in most captions, so the trigger learns them.
      Captions the user brought never change.
    """
    check_project_name(project_name)
    trigger_word = (trigger_word or "").strip()
    route = registry.get(model)
    if auto_tag is None:
        auto_tag = route.default_auto_tag
    custom = _check_quick_inputs(route, dataset, base, convrot_int8, plan_kwargs)
    # Drive when mounted: Colab wipes /content, and every saved epoch, when the session ends.
    workspace = Path(workspace) / project_name if workspace else environment.default_workspace() / project_name

    dataset_dir = _ready_dataset(dataset, project_name, trigger_word, auto_tag, prune_appearance_tags, plan_kwargs)

    base_spec, checkpoint, vae, text_encoder = models.prepare_base(
        route,
        None if custom else base,
        custom=custom,
        convrot_int8=convrot_int8,
        config_overrides=plan_kwargs.get("config_overrides"),
    )
    plan = build_plan(
        route,
        base_spec,
        checkpoint_path=checkpoint,
        vae_path=vae,
        text_encoder_path=text_encoder,
        dataset_dir=dataset_dir,
        workspace=workspace,
        project_name=project_name,
        trigger_word=trigger_word,
        **plan_kwargs,
    )
    if not dry_run:
        execute(plan)
    return plan

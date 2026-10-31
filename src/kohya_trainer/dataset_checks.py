"""Dataset settings and captions: where each setting goes, the checks that
refuse a dataset with nothing to train on, and the step estimate."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

from . import config as config_mod
from . import dataset as dataset_mod
from .registry import ModelRoute

# The backend ignores these when a dataset config is given (train_network.py warns and goes on),
# and build_plan always gives one.
IGNORED_DATASET_ARGS = ("train_data_dir", "reg_data_dir", "in_json")

# Training arguments the dataset config also sets, by their dataset-config
# name. The dataset config wins in the backend (library/config_util.py), so these move there.
DATASET_ARGS = {
    **{key: key for key in sorted(dataset_mod.GENERAL_KEYS)},
    "train_batch_size": "batch_size",
    "dataset_repeats": "num_repeats",
}


def check_dataset_args(config_overrides: dict | None) -> None:
    """Refuse dataset arguments in config_overrides that the backend ignores."""
    found = [key for key in IGNORED_DATASET_ARGS if (config_overrides or {}).get(key) is not None]
    if found:
        raise ValueError(
            f"the trainer ignores {', '.join(found)} in config_overrides, because it reads the dataset from "
            "the dataset config that this notebook writes. Point the dataset field at your image folder instead, and "
            "use subfolders named like 10_name to set the repeats."
        )


# Dataset settings (library/config_util.py) that build_dataset_config does not write.
# The backend reads them from the training arguments for every folder, so they move
# to config_overrides. A test checks them against the pin.
DATASET_FALLBACK_ARGS = frozenset(
    {
        "alpha_mask",
        "cache_info",
        "caption_dropout_every_n_epochs",
        "caption_dropout_rate",
        "caption_separator",
        "caption_suffix",
        "caption_tag_dropout_rate",
        "color_aug",
        "enable_multiline_caption",
        "enable_wildcard",
        "face_crop_aug_range",
        "keep_tokens_separator",
        "multiline_caption_separator",
        "random_crop",
        "resize_interpolation",
        "secondary_separator",
        "skip_image_resolution",
        "token_warmup_min",
        "token_warmup_step",
        "validation_seed",
        "validation_split",
    }
)


def dataset_override_keys() -> frozenset[str]:
    """The keys dataset_overrides passes to build_dataset_config."""
    params = inspect.signature(dataset_mod.build_dataset_config).parameters
    return frozenset(name for name in params if name not in ("image_dir", "folder_class_tokens"))


def split_dataset_overrides(dataset_overrides: dict | None) -> tuple[dict, dict, list[str]]:
    """Check dataset_overrides and sort it.

    Returns the settings for build_dataset_config, the ones that move to
    config_overrides, and notes that say what moved. An unknown key is refused.
    """
    accepted = dataset_override_keys()
    given = dict(dataset_overrides or {})
    kept: dict = {}
    moved: dict = {}
    unknown: list[str] = []
    notes: list[str] = []
    for key, value in given.items():
        if key == "train_batch_size":
            if "batch_size" not in given:
                kept["batch_size"] = value
            notes.append(
                "train_batch_size in dataset_overrides is batch_size in the dataset config, so it is used as that."
            )
        elif key in accepted:
            kept[key] = value
        elif key in DATASET_FALLBACK_ARGS:
            moved[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"dataset_overrides cannot take {', '.join(sorted(unknown))}. It takes {', '.join(sorted(accepted))}, "
            "and dataset settings such as caption_dropout_rate or color_aug. Put training settings in "
            "config_overrides."
        )
    if moved:
        notes.append(
            f"{', '.join(moved)} in dataset_overrides moved to config_overrides. The dataset config this notebook "
            "writes has no place for them, and the trainer reads them from there for every folder."
        )
    return kept, moved, notes


def dataset_settings(config_overrides: dict, dataset_overrides: dict | None) -> dict:
    """Move dataset settings out of config_overrides, in place, and return the dataset overrides.

    A key in both keeps the dataset_overrides value.
    """
    merged = dict(dataset_overrides or {})
    moved: list[str] = []
    shadowed: list[str] = []
    for arg, key in DATASET_ARGS.items():
        if arg not in config_overrides:
            continue
        value = config_overrides.pop(arg)
        if value is None:
            continue
        if key == "resolution" and isinstance(value, str):
            # The command-line form, "768" or "768,1024".
            sizes = [int(part) for part in value.split(",")]
            value = sizes[0] if len(sizes) == 1 else sizes
        moved.append(arg)
        if key in merged:
            shadowed.append(key)
        merged.setdefault(key, value)
    if moved:
        print(
            f"note: {', '.join(moved)} in config_overrides moved to the dataset settings, where the trainer "
            "reads them."
            + (f" dataset_overrides also sets {', '.join(shadowed)}, and its value is used." if shadowed else "")
        )
    return merged


def caption_sources(
    subsets: list[dict], trigger_word: str, class_tokens: str | None, extra_text: str | None = None
) -> dict:
    """Count what each image trains on, over every subset (``dataset.subset_counts``).

    An image without a caption file trains on the trigger word, else on its
    folder's name, class_tokens or ``extra_text``, else on an empty caption.
    quick_train and build_plan count this way, so they refuse the same datasets.
    """
    counts = {"captioned": 0, "trigger": 0, "class_tokens": 0, "empty": 0}
    for subset in subsets:
        bare = subset["images"] - subset["captioned"]
        counts["captioned"] += subset["captioned"]
        if trigger_word:
            counts["trigger"] += bare
        elif subset["name"] or class_tokens or (extra_text or "").strip():
            counts["class_tokens"] += bare
        else:
            counts["empty"] += bare
    return counts


def _extra_caption_text(dataset_overrides: dict, config_overrides: dict) -> str:
    """A caption_prefix or caption_suffix the user set, in either dict, or ""."""
    for key in ("caption_prefix", "caption_suffix"):
        for scope in (dataset_overrides, config_overrides):
            value = str(scope.get(key) or "").strip()
            if value:
                return value
    return ""


def _refuse_no_captions(sources: dict, fix: str) -> None:
    """Refuse a dataset where no image has a caption, a trigger or a class token. ``fix`` says what to do."""
    if not (sources["captioned"] or sources["trigger"] or sources["class_tokens"]):
        raise ValueError(
            f"the dataset has no captions and no trigger_word. Training on empty captions ruins the model. {fix}"
        )


def _refuse_no_images(dataset_dir: Path, summary: dict, dataset_field: str) -> None:
    """Refuse a dataset with no image the trainer reads. ``dataset_field`` names the field to change."""
    if summary["images"]:
        return
    hint = (
        f". {summary['nested_images']} images are in subfolders, but the trainer only reads the folder "
        f"itself and subfolders named like 10_name. Point {dataset_field} at the folder the images are in."
        if summary["nested_images"]
        else ""
    )
    raise FileNotFoundError(f"no images found under {dataset_dir}{hint}")


def _caption_extension(dataset_overrides: dict | None, config_overrides: dict | None) -> str:
    """The caption extension the trainer reads: dataset_overrides, then config_overrides, then .txt."""
    return (
        (dataset_overrides or {}).get("caption_extension")
        or (config_overrides or {}).get("caption_extension")
        or ".txt"
    )


def route_training_defaults(route: ModelRoute) -> dict:
    """The route's dataset defaults that go to the training settings, below the user's values."""
    return {key: value for key, value in route.dataset_defaults if key in DATASET_FALLBACK_ARGS}


def check_caption_candidates(train_config: dict, dataset_config: dict, entrypoint: str) -> None:
    """Refuse multiline captions where the text cache keeps only one, and bad separator lines.

    Both stop the trainer before it loads a model. A bad separator line stops it
    while it reads the captions. Several candidates stop it in its argument
    checks (assert_extra_args).
    """
    bad = dataset_mod.bad_separator_lines(dataset_config, train_config)
    if bad:
        image, line = bad[0]
        raise ValueError(
            f"the caption for {image.name} has a separator line with text after it ({line!r}), and the trainer "
            "stops on that while loading the dataset: only a # comment can follow the separator. Move that "
            "text to the next line, or start it with #."
        )
    caches = train_config.get("cache_text_encoder_outputs") or train_config.get("cache_text_encoder_outputs_to_disk")
    if not caches or entrypoint in config_mod.TE_CACHE_KEEPS_CANDIDATES:
        return
    found = dataset_mod.captions_with_candidates(dataset_config, train_config)
    if found:
        separator = train_config.get("multiline_caption_separator") or dataset_mod.DEFAULT_MULTILINE_CAPTION_SEPARATOR
        raise ValueError(
            f"{len(found)} caption file{'s' if len(found) != 1 else ''} (such as the one for {found[0].name}) "
            f"hold several captions split by {separator} lines. This model's text-encoder cache keeps only "
            "one caption per image, so the trainer stops before it loads the model. Remove the separator lines, turn "
            "off enable_multiline_caption, or turn text caching off by adding 'cache_text_encoder_outputs': None, "
            "'cache_text_encoder_outputs_to_disk': None to config_overrides (slower, and it uses more GPU memory)."
        )


def _check_dataset(
    dataset_dir: Path,
    run_overrides: dict,
    dataset_overrides: dict,
    trigger_word: str,
    num_repeats: int | None,
) -> tuple[list[dict], int]:
    """Count the dataset, refuse one with nothing to train on, and print its notes.

    Returns its subset counts and its number of images.
    """
    caption_extension = dataset_overrides.get("caption_extension", ".txt")
    layout = dataset_mod.find_subsets(dataset_dir)
    summary = dataset_mod.summarize(dataset_dir, caption_extension)
    subsets = dataset_mod.subset_counts(dataset_dir, caption_extension)
    _refuse_no_images(dataset_dir, summary, "the dataset")
    for note in dataset_mod.layout_notes(layout):
        print(f"note: {note}")
    if summary["captioned"] == 0 and caption_extension != ".txt":
        txt = dataset_mod.summarize(dataset_dir, ".txt")["captioned"]
        if txt:
            print(
                f"note: no image has a {caption_extension} caption file, but {txt} have a .txt one. "
                f"caption_extension is {caption_extension}, so the trainer reads none of them. Change it in "
                "dataset_overrides, or tag again with the same caption_extension."
            )
    sources = caption_sources(
        subsets,
        trigger_word,
        dataset_overrides.get("class_tokens"),
        _extra_caption_text(dataset_overrides, run_overrides),
    )
    _refuse_no_captions(sources, "Add .txt caption files (or run the tagger), or set a trigger_word.")
    if sources["empty"]:
        print(
            f"note: {sources['empty']} images have no caption file and nothing else to train on, so they train "
            "on an empty caption. Add caption files for them, or set a trigger_word."
        )
    if sources["trigger"] and sources["captioned"]:
        print(f"note: {sources['trigger']} images have no caption file and will train on the trigger word only.")
    if str(dataset_dir).replace("\\", "/").startswith("/content/drive/"):
        # A zip is already on local disk. A Drive folder trains in place.
        print(
            "note: this dataset folder is on Google Drive. Training reads the latent caches next to the "
            "images every epoch, which is slow on Drive and uses your Drive storage. A .zip on Drive is "
            "extracted to fast local disk instead. A folder keeps its caches for the next session."
        )
    if layout.numbered and (num_repeats is not None or "num_repeats" in dataset_overrides):
        rest = " num_repeats is used for the images in the top folder." if layout.subsets[0].num_repeats is None else ""
        print(
            "note: num_repeats is set, but the folder names set the repeats (10_mychar trains 10 times per "
            f"epoch), so the folder numbers are used.{rest}"
        )
    return subsets, summary["images"]


def _step_estimate(train_config: dict, dataset_config: dict, subsets: list[dict], batch: int) -> tuple[int, int | None]:
    """Print and return a rough count of steps per epoch and in all. Bucketing changes it a little."""
    subset_configs = dataset_config["datasets"][0]["subsets"]
    counts = {str(Path(s["image_dir"]).resolve()): s["images"] for s in subsets}
    per_subset = [(Path(s["image_dir"]).name, counts.get(s["image_dir"], 0), s["num_repeats"]) for s in subset_configs]
    accumulation = train_config.get("gradient_accumulation_steps") or 1
    per_epoch = sum(count * repeats for _, count, repeats in per_subset)
    steps_per_epoch = math.ceil(per_epoch / max(1, batch * accumulation))
    epochs = train_config.get("max_train_epochs")
    total_steps = steps_per_epoch * epochs if epochs else train_config.get("max_train_steps")
    if len(per_subset) == 1:
        images = f"{per_subset[0][1]} images x {per_subset[0][2]} repeats"
    else:
        images = " + ".join(f"{name} {count} x {repeats}" for name, count, repeats in per_subset)
    print(
        f"planned run: {images} / batch {batch} "
        f"= about {steps_per_epoch} steps per epoch, about {total_steps} steps"
        + (f" over {epochs} epochs" if epochs else "")
    )
    return steps_per_epoch, total_steps

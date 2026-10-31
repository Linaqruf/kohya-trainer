"""The WD tagger, run from the pinned submodule (finetune/tag_images_by_wd14_tagger.py).

We keep no copy of the tagger, so tagging always matches the pin. onnxruntime
is installed only when tagging runs.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import dataset, environment
from .dataset import normalize_tag

# The v3 taggers come only as ONNX. EVA02-large, the default, is the most accurate.
TAGGERS = (
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
)
DEFAULT_TAGGER = TAGGERS[0]

# The trainers that offer use_rating_tags. Anima rates images with other words
# (safe, sensitive, nsfw, explicit), so WD's rating tags teach it the wrong ones.
RATING_TAG_ENTRYPOINTS = ("train_network.py", "sdxl_train_network.py")


def _tag_list(tags: str, normalize: bool) -> str:
    items = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return ",".join(normalize_tag(tag) if normalize else tag for tag in items)


def build_tag_command(
    image_dir: str | Path,
    *,
    sd_scripts_dir: str | Path,
    repo_id: str = DEFAULT_TAGGER,
    general_threshold: float = 0.35,
    character_threshold: float = 0.35,
    caption_extension: str = ".txt",
    batch_size: int = 8,
    undesired_tags: str = "",
    remove_underscore: bool = True,
    append_tags: bool = False,
    character_tags_first: bool = False,
    use_rating_tags: bool = False,
    # sd-scripts does not train images in subfolders.
    recursive: bool = False,
) -> list[str]:
    """The tagger command line for ``image_dir``."""
    script = Path(sd_scripts_dir) / "finetune" / "tag_images_by_wd14_tagger.py"
    cmd = [
        sys.executable,
        str(script),
        str(image_dir),
        "--onnx",
        f"--repo_id={repo_id}",
        f"--general_threshold={general_threshold}",
        f"--character_threshold={character_threshold}",
        f"--caption_extension={caption_extension}",
        f"--batch_size={batch_size}",
        "--max_data_loader_n_workers=2",
    ]
    # The tagger matches undesired_tags after remove_underscore, so they get the same change.
    undesired = _tag_list(undesired_tags, remove_underscore)
    if undesired:
        cmd.append(f"--undesired_tags={undesired}")
    if remove_underscore:
        cmd.append("--remove_underscore")
    if append_tags:
        cmd.append("--append_tags")
    if character_tags_first:
        cmd.append("--character_tags_first")
    if use_rating_tags:
        cmd.append("--use_rating_tags")
    if recursive:
        cmd.append("--recursive")
    return cmd


def tag_images(
    image_dir: str | Path,
    sd_scripts_dir: str | Path | None = None,
    backup_dir: str | Path | None = None,
    **options,
) -> None:
    """Write a caption file for each training image in ``image_dir`` with the WD tagger.

    Each subset folder is tagged in turn. With ``backup_dir`` and
    ``append_tags`` off, the old caption files are first copied to a dated
    folder in it.
    """
    sd_scripts_dir = sd_scripts_dir or environment.sd_scripts_dir()
    environment.ensure_tagger_deps()
    if backup_dir and not options.get("append_tags"):
        _back_up_captions(Path(image_dir), Path(backup_dir), options.get("caption_extension", ".txt"))
    folders = [s.image_dir for s in dataset.find_subsets(image_dir).subsets if dataset.image_files(s.image_dir)]
    for folder in folders or [Path(image_dir)]:
        if len(folders) > 1:
            print(f"tagging {folder.name}")
        cmd = build_tag_command(folder, sd_scripts_dir=sd_scripts_dir, **options)
        returncode, tail = environment.run_streamed(cmd, cwd=sd_scripts_dir)
        if returncode != 0:
            raise RuntimeError(f"tagging failed: the tagger exited with code {returncode}. Tagger output tail:\n{tail}")


def _back_up_captions(image_dir: Path, backup_dir: Path, caption_extension: str) -> Path | None:
    files = dataset.caption_files(image_dir, caption_extension)
    if not files:
        return None
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = backup_dir / stamp
    suffix = 2
    while dest.exists():  # two runs in the same second
        dest = backup_dir / f"{stamp}_{suffix}"
        suffix += 1
    for caption in files:
        target = dest / caption.parent.relative_to(image_dir) / caption.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(caption, target)
    print(f"copied {len(files)} caption files to {dest} before the tagger replaces them")
    return dest


def remove_tags(image_dir: str | Path, tags: str, caption_extension: str = ".txt") -> int:
    """Remove these tags (a comma list, with or without underscores) from every caption file.

    Returns how many caption files changed.
    """
    unwanted = {normalize_tag(tag) for tag in tags.split(",") if tag.strip()}
    if not unwanted:
        return 0
    changed, removed = _strip_tags(image_dir, unwanted, caption_extension)
    if changed:
        print(f"removed {removed} tags from {changed} caption files")
    else:
        print(f"none of these tags are in the captions: {', '.join(sorted(unwanted))}")
    return changed


def _strip_tags(image_dir: str | Path, unwanted: set[str], caption_extension: str) -> tuple[int, int]:
    """Remove the normalized ``unwanted`` tags. Returns the files changed and the tags removed."""
    changed = removed = 0
    for caption in dataset.caption_files(image_dir, caption_extension):
        text = caption.read_text(encoding="utf-8", errors="replace")
        lines = []
        for line in text.split("\n"):
            kept = [tag.strip() for tag in line.split(",") if tag.strip()]
            left = [tag for tag in kept if normalize_tag(tag) not in unwanted]
            removed += len(kept) - len(left)
            lines.append(", ".join(left) if len(left) != len(kept) else line)
        new_text = "\n".join(lines)
        if new_text != text:
            caption.write_text(new_text, encoding="utf-8", newline="\n")
            changed += 1
    return changed, removed


# Hair and eye tags that describe the subject. Pose tags such as closed eyes stay out.
_COLOURS = (
    "black",
    "white",
    "grey",
    "brown",
    "blonde",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "aqua",
    "purple",
    "pink",
    "silver",
)
_SHADES = tuple(f"{shade} {colour}" for shade in ("light", "dark") for colour in _COLOURS)
APPEARANCE_TAGS = frozenset(
    [f"{colour} hair" for colour in (*_COLOURS, *_SHADES, "platinum blonde")]
    + [f"{colour} eyes" for colour in (*_COLOURS, *_SHADES)]
    + ["multicolored hair", "two-tone hair", "gradient hair", "streaked hair", "colored inner hair", "split-color hair"]
    + ["multicolored eyes", "heterochromia"]
    + ["very short hair", "short hair", "medium hair", "long hair", "very long hair", "absurdly long hair"]
)


def prune_into_trigger(image_dir: str | Path, caption_extension: str = ".txt", min_share: float = 0.5) -> list[str]:
    """Remove the APPEARANCE_TAGS in at least ``min_share`` of the captions, so the trigger word learns them.

    Returns them, most common first, as the captions spell them.
    """
    counts, files, _ = dataset.tag_counts(image_dir, caption_extension)
    if not files:
        return []
    picked = [
        (tag, n) for tag, n in counts.most_common() if normalize_tag(tag) in APPEARANCE_TAGS and n >= min_share * files
    ]
    if not picked:
        print("no hair or eye tag is in most captions, so none were taken out")
        return []
    _strip_tags(image_dir, {normalize_tag(tag) for tag, _ in picked}, caption_extension)
    named = ", ".join(f"{tag} ({n} of {files})" for tag, n in picked)
    print(f"took {named} out of the captions, so the trigger word learns them")
    return [tag for tag, _ in picked]

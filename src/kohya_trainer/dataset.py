"""Dataset preparation and the sd-scripts dataset-config TOML builder.

A dataset is a folder of images with ``.txt`` caption files of the same name,
and can hold ``<N>_<name>`` subfolders (``10_mychar``) that train N times per
epoch. The schema in library/config_util.py refuses unknown keys, so the
builder writes only the keys in the sets below. A test checks them against the pin.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# From library/config_util.py. A subset value wins over the dataset's, then general, then the command line.
GENERAL_KEYS = frozenset(
    {
        "resolution",
        "enable_bucket",
        "min_bucket_reso",
        "max_bucket_reso",
        "bucket_reso_steps",
        "bucket_no_upscale",
        "caption_extension",
        "shuffle_caption",
        "keep_tokens",
        "flip_aug",
    }
)
DATASET_KEYS = frozenset({"batch_size"})
SUBSET_KEYS = frozenset({"image_dir", "num_repeats", "class_tokens", "caption_prefix"})

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Each trainer saves an image's text-encoder cache next to it as "<image stem><suffix>"
# (library/strategy_sdxl.py, library/strategy_anima.py).
TEXT_CACHE_SUFFIXES = {
    "sdxl_train_network.py": "_te_outputs.npz",
    "anima_train_network.py": "_anima_te.npz",
    "moss_image_train_network.py": "_moss_te.npz",
}

# Training arguments that change what the text-encoder cache holds, per trainer.
# On SDXL and Anima, mixed_precision sets the dtype the text encoders encode in.
# On MOSS-Image, the bf16 and ConvRot INT8 text encoders encode differently.
# MOSS-Image's own cache check compares only the caption texts.
TEXT_CACHE_ARGS = {
    "sdxl_train_network.py": (
        "pretrained_model_name_or_path",
        "max_token_length",
        "weighted_captions",
        "mixed_precision",
    ),
    "anima_train_network.py": (
        "qwen3",
        "t5_tokenizer_path",
        "qwen3_max_token_length",
        "t5_max_token_length",
        "mixed_precision",
    ),
    "moss_image_train_network.py": (
        "text_encoder",
        "convrot_int8_text_encoder",
        "moss_max_token_length",
    ),
}
_MODEL_FILE_ARGS = frozenset({"pretrained_model_name_or_path", "qwen3", "t5_tokenizer_path", "text_encoder"})

# Caption settings the cached text depends on, with the backend's defaults
# (library/config_util.py). The Anima and MOSS-Image caches also store caption_dropout_rate.
_CAPTION_SETTINGS = {
    "caption_extension": ".caption",
    "class_tokens": None,
    "caption_prefix": None,
    "caption_suffix": None,
    "caption_separator": ",",
    "secondary_separator": None,
    "enable_wildcard": False,
    "enable_multiline_caption": False,
    "multiline_caption_separator": None,
    "caption_dropout_rate": 0.0,
}
_JOINED_SETTINGS = frozenset({"class_tokens", "caption_prefix", "caption_suffix"})
# The backend's default multiline_caption_separator (library/subset.py).
DEFAULT_MULTILINE_CAPTION_SEPARATOR = "---===---"

# Dotfiles, which the backend never reads as an image or a caption.
TEXT_CACHE_RECORD = ".kohya-trainer-text-cache.json"
EXTRACT_RECORD = ".kohya-trainer-extracted.json"


SOURCE_FORMS = (
    "a folder, a .zip file, an https link to a .zip file, or a Hugging Face dataset repo of image files "
    "or one .zip (user/name)"
)
# Groups: the repo, a /tree/ link's revision and folder, a /resolve/ or /blob/ link's revision and file.
_HF_DATASET_URL = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/datasets/([\w.-]+/[\w.-]+)"
    r"(?:/tree/([^/?#]+)(?:/([^?#]*?))?|/(?:resolve|blob)/([^/]+)/(.+?))?/?(?:[?#].*)?$",
    re.IGNORECASE,
)


class BadArchive(ValueError):
    """A .zip file that does not open, or does not read back whole."""


_REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")


def prepare_dataset(source: str | Path, workspace: Path) -> Path:
    """Return the absolute path of an image folder ready to train.

    ``source`` is a folder, a .zip file, an https link to a .zip file, or a
    Hugging Face dataset (``user/name`` or its link). A local path wins.
    Downloads and extracted files go in ``workspace``. The trainer and the
    tagger run from the sd-scripts folder, so a relative path would point to the
    wrong place.
    """
    text = str(source).strip()
    path = Path(text).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.suffix.lower() == ".zip" and path.is_file():
        return _extract(path, workspace / "dataset")
    if text.lower().startswith(("https://", "http://")):
        return _from_link(text, workspace)
    if _REPO_ID.fullmatch(text) and not path.exists():
        return _from_hub(text, workspace)
    hint = ""
    if text.startswith("/content/drive") and not Path("/content/drive").is_dir():
        hint = ". Google Drive is not mounted. Tick mount_google_drive, or run environment.mount_drive()."
    raise FileNotFoundError(f"dataset not found: {text}. It can be {SOURCE_FORMS}{hint}")


def _from_link(url: str, workspace: Path) -> Path:
    from urllib.parse import unquote, urlparse

    hub = _HF_DATASET_URL.match(url)
    if hub and not hub.group(5):
        revision = unquote(hub.group(2)) if hub.group(2) else None
        subfolder = unquote(hub.group(3) or "").strip("/")
        return _from_hub(hub.group(1), workspace, revision=revision, subfolder=subfolder)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "drive.google.com" in host or "drive.usercontent.google.com" in host:
        raise ValueError(
            "a Google Drive share link cannot be downloaded here. Tick mount_google_drive and give the file's "
            "path instead, such as /content/drive/MyDrive/dataset.zip."
        )
    name = Path(parsed.path).name
    if not name.lower().endswith(".zip"):
        raise ValueError(f"{url} is not a link to a .zip file. The dataset can be {SOURCE_FORMS}.")
    # One folder per link, so two links to files of the same name never mix.
    target = workspace / "downloads" / hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] / name
    if hub:
        from huggingface_hub import hf_hub_download

        repo, revision, filename = hub.group(1), hub.group(4), hub.group(5)
        target = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=target.parent))
    else:
        from . import models

        models.download_file(url, target)
    try:
        return _extract(target, workspace / "dataset")
    except BadArchive as error:
        target.unlink(missing_ok=True)
        raise ValueError(
            f"the link {url} did not give a working .zip file ({error}). It probably leads to a web page, not "
            "the file. The download was deleted, so the next run downloads it again. Use a direct "
            "download link to the .zip."
        ) from error


def _from_hub(repo_id: str, workspace: Path, revision: str | None = None, subfolder: str = "") -> Path:
    """Fetch a Hugging Face dataset of image files once, and keep that copy and its captions.

    A single .zip at the top is extracted. ``revision`` and ``subfolder`` come from a /tree/ link.
    """
    name = repo_id.replace("/", "--")
    if revision or subfolder:
        name += "--" + hashlib.sha256(f"{revision}/{subfolder}".encode()).hexdigest()[:8]
    folder = workspace / "hub" / name
    record_path = folder.parent / f".{folder.name}.json"
    record: dict = {"repo": repo_id}
    if revision:
        record["revision"] = revision
    if subfolder:
        record["folder"] = subfolder
    where = repo_id + (f" at {revision}" if revision else "")
    if folder.is_dir() and _read_record(record_path) == record:
        print(
            f"using the copy of {where} fetched earlier, with any captions made since. To fetch it again, "
            f"delete {folder}."
        )
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError

        record_path.unlink(missing_ok=True)
        options: dict = {}
        if revision:
            options["revision"] = revision
        if subfolder:
            options["allow_patterns"] = f"{subfolder}/*"
        try:
            snapshot_download(repo_id, repo_type="dataset", local_dir=folder, **options)
        except RepositoryNotFoundError as error:
            raise FileNotFoundError(
                f"dataset not found: no folder or file {repo_id!r} here, and no Hugging Face dataset "
                f"{repo_id} you can read. A private dataset needs an HF_TOKEN secret with read access. "
                f"The dataset can be {SOURCE_FORMS}."
            ) from error
        record_path.write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
    root = folder / subfolder if subfolder else folder
    if not root.is_dir():
        raise FileNotFoundError(f"the folder {subfolder} is not in the Hugging Face dataset {where}")
    _check_hub_format(root, where)
    zips = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    if len(zips) == 1:
        return _extract(zips[0], workspace / "dataset")
    if subfolder and repeat_folder(root.name):
        # The folder's name holds its repeats, so its parent is the dataset.
        return _descend(root.parent)
    return _descend(root)


def _check_hub_format(root: Path, where: str) -> None:
    """Refuse a Hub dataset that keeps its images inside data files (.parquet or metadata.jsonl)."""
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(_skipped_folder(part) for part in p.relative_to(root).parts[:-1])
    ]
    if any(_is_image(p) or p.suffix.lower() == ".zip" for p in files):
        if any(p.name == "metadata.jsonl" for p in files):
            print(
                f"note: the Hugging Face dataset {where} keeps captions in metadata.jsonl, which the trainer "
                "does not read. It reads only caption files next to the images, so images without one train "
                "as if they had no caption."
            )
        return
    if any(p.suffix.lower() == ".parquet" or p.name == "metadata.jsonl" for p in files):
        raise ValueError(
            f"the Hugging Face dataset {where} has no image files, only .parquet or metadata.jsonl data files. "
            "That format is not supported here. Use a dataset repo of image files (or one .zip), or download "
            "the images and give their folder."
        )


def _extract(archive: Path, dest: Path) -> Path:
    """Extract a zip to ``dest``, or keep what an earlier run extracted from it.

    The same zip (path, size and time) keeps the folder with its captions and
    caches. A changed zip is extracted fresh.
    """
    stat = archive.stat()
    current = {"zip": str(archive.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    record_path = dest.parent / EXTRACT_RECORD
    if dest.is_dir() and _read_record(record_path) == current:
        print(
            f"{archive.name} has not changed since it was extracted, so the extracted folder is kept, with "
            "any captions made since. Change the zip to extract it again."
        )
        return _descend(dest)
    # Opened before anything is removed, so a bad file never costs the old folder.
    try:
        with zipfile.ZipFile(archive) as opened:
            opened.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        raise BadArchive(f"{archive.name} is not a zip file that opens") from error
    # Swapped in only once complete, so a cut-off run never leaves half a dataset.
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(partial)
    except (zipfile.BadZipFile, EOFError, zlib.error) as error:
        shutil.rmtree(partial, ignore_errors=True)
        raise BadArchive(f"{archive.name} is damaged and does not extract ({error})") from error
    except (RuntimeError, NotImplementedError) as error:
        # A password, or a compression such as Deflate64.
        shutil.rmtree(partial, ignore_errors=True)
        raise BadArchive(
            f"{archive.name} needs a password or uses a compression this runtime cannot read ({error}). "
            "Make a plain zip of the images and try again."
        ) from error
    except OSError:
        # A full disk is not the zip's fault, so the zip is kept.
        shutil.rmtree(partial, ignore_errors=True)
        raise
    record_path.unlink(missing_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    record_path.write_text(json.dumps(current) + "\n", encoding="utf-8", newline="\n")
    return _descend(dest)


def _descend(folder: Path) -> Path:
    """The folder to train from, when an archive or a repo wraps everything in one folder.

    A single ``<N>_<name>`` folder is not entered, because its name holds the repeats.
    """
    entries = [p for p in folder.iterdir() if not _skipped_folder(p.name) and (p.is_dir() or _is_image(p))]
    if len(entries) == 1 and entries[0].is_dir() and repeat_folder(entries[0].name) is None:
        return entries[0].resolve()
    return folder.resolve()


_IMAGE_EXTENSIONS_UPPER = {ext.upper() for ext in IMAGE_EXTENSIONS}


def _is_image(path: Path) -> bool:
    """Match images as sd-scripts does: *.ext and *.EXT, never mixed case or dotfiles.

    .avif and .jxl are left out: they need Pillow plugins that are not installed.
    """
    if path.name.startswith("."):
        return False
    return path.suffix in IMAGE_EXTENSIONS or path.suffix in _IMAGE_EXTENSIONS_UPPER


# The same names the backend reads from train_data_dir
# (generate_dreambooth_subsets_config_by_subdirs in library/config_util.py).
_REPEAT_FOLDER = re.compile(r"(\d+)_(.*)")
# A folder number above this gets a note: it can be a date, as in 2024_trip.
LARGE_REPEATS = 100


@dataclass(frozen=True)
class Subset:
    """One image folder that trains as a subset, with the repeats and name of a ``<N>_<name>`` folder."""

    image_dir: Path
    num_repeats: int | None = None
    name: str = ""


@dataclass(frozen=True)
class Layout:
    """The subsets in a dataset folder, and the subfolders that do not train."""

    root: Path
    subsets: tuple[Subset, ...]
    # Subfolders with images that are not named <N>_<name>.
    ignored: tuple[str, ...] = ()
    # <N>_<name> folders with no images in them.
    empty: tuple[str, ...] = ()

    @property
    def numbered(self) -> bool:
        """True when some repeats come from folder names."""
        return any(subset.num_repeats for subset in self.subsets)


def _skipped_folder(name: str) -> bool:
    # macOS zips carry a __MACOSX folder. Dot folders hold tool data (.ipynb_checkpoints, .cache).
    return name.startswith((".", "__MACOSX"))


def repeat_folder(name: str) -> tuple[int, str] | None:
    """(N, name) for a folder named ``<N>_<name>`` with N of 1 or more, else None."""
    match = _REPEAT_FOLDER.fullmatch(name)
    if not match or int(match.group(1)) < 1:
        return None
    return int(match.group(1)), match.group(2)


def image_files(folder: str | Path) -> list[Path]:
    """The images the trainer reads in ``folder``, not in its subfolders."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*") if _is_image(p))


def find_subsets(dataset_dir: str | Path) -> Layout:
    """The subsets the dataset trains as: each ``<N>_<name>`` folder with images, and the top folder's images.

    Without a ``<N>_<name>`` folder that holds images, the top folder is the one
    subset. Other subfolders are not read.
    """
    root = Path(dataset_dir)
    numbered: list[Subset] = []
    ignored: list[str] = []
    empty: list[str] = []
    children = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for child in children:
        if _skipped_folder(child.name):
            continue
        parsed = repeat_folder(child.name)
        if parsed is None:
            if any(_is_image(p) for p in child.rglob("*")):
                ignored.append(child.name)
        elif image_files(child):
            numbered.append(Subset(child, *parsed))
        else:
            empty.append(child.name)
    subsets = list(numbered)
    if image_files(root) or not subsets:
        subsets.insert(0, Subset(root))
    return Layout(root, tuple(subsets), tuple(ignored), tuple(empty))


def layout_notes(layout: Layout) -> list[str]:
    """Plain notes about what in the folder trains and what does not."""
    notes = []
    if layout.numbered and layout.subsets[0].num_repeats is None:
        count = len(image_files(layout.root))
        images = "1 image is" if count == 1 else f"{count} images are"
        notes.append(
            f"{images} in the top folder next to the numbered folders. They train as one more subset, with num_repeats."
        )
    if layout.empty:
        notes.append(f"these folders have no images, so they are left out: {', '.join(layout.empty)}")
    if layout.ignored:
        notes.append(
            "these subfolders are not trained, because their names do not start with a number of 1 or more "
            f"and an underscore (such as 10_mychar): {', '.join(layout.ignored)}"
        )
    for subset in layout.subsets:
        if subset.num_repeats and subset.num_repeats > LARGE_REPEATS:
            notes.append(
                f"{subset.image_dir.name} trains {subset.num_repeats} times per epoch, from the number in its "
                "name. If that number is a date or a count, not repeats, rename the folder, such as "
                f"10_{subset.name or 'name'}."
            )
    return notes


def subset_counts(image_dir: str | Path, caption_extension: str = ".txt") -> list[dict]:
    """For each subset: its folder, repeats and name, and its counts of images and caption files."""
    counts = []
    for subset in find_subsets(image_dir).subsets:
        images = image_files(subset.image_dir)
        counts.append(
            {
                "image_dir": str(subset.image_dir),
                "folder": subset.image_dir.name,
                "num_repeats": subset.num_repeats,
                "name": subset.name,
                "images": len(images),
                "captioned": sum(1 for p in images if p.with_suffix(caption_extension).is_file()),
            }
        )
    return counts


def summarize(image_dir: Path, caption_extension: str = ".txt") -> dict:
    """Count the images the trainer reads and their captions, and the images no subset reads.

    A subset folder is read without its subfolders (glob_images in library/dataset.py).
    """
    root = Path(image_dir)
    subsets = subset_counts(root, caption_extension)
    total = sum(s["images"] for s in subsets)
    captioned = sum(s["captioned"] for s in subsets)
    every_image = sum(1 for p in root.rglob("*") if _is_image(p)) if root.is_dir() else 0
    return {
        "images": total,
        "captioned": captioned,
        "uncaptioned": total - captioned,
        "nested_images": every_image - total,
    }


def describe(image_dir: str | Path, caption_extension: str = ".txt") -> str:
    """The dataset summary in a few plain lines, with the notes about its folders."""
    summary = summarize(Path(image_dir), caption_extension)
    subsets = subset_counts(image_dir, caption_extension)
    lines = [f"{summary['images']} images, {summary['captioned']} with a {caption_extension} caption file"]
    if len(subsets) > 1 or subsets[0]["num_repeats"]:
        for subset in subsets:
            repeats = f"{subset['num_repeats']} repeats" if subset["num_repeats"] else "num_repeats from the form"
            lines.append(f"  {subset['folder']}: {subset['images']} images, {subset['captioned']} captioned, {repeats}")
    if summary["nested_images"]:
        lines.append(f"{summary['nested_images']} more images are in subfolders the trainer does not read")
    lines += [f"note: {note}" for note in layout_notes(find_subsets(image_dir))]
    return "\n".join(lines)


def caption_files(image_dir: str | Path, caption_extension: str = ".txt") -> list[Path]:
    """The caption file of each training image that has one, in every subset."""
    files = []
    for subset in find_subsets(image_dir).subsets:
        for image in image_files(subset.image_dir):
            caption = image.with_suffix(caption_extension)
            if caption.is_file():
                files.append(caption)
    return files


def tag_report(image_dir: str | Path, caption_extension: str = ".txt", top: int = 24) -> str:
    """A report of how often each tag appears in the caption files, to help clean them up."""
    counts: Counter[str] = Counter()
    files = caption_files(image_dir, caption_extension)
    for caption in files:
        for tag in caption.read_text(encoding="utf-8", errors="replace").split(","):
            if tag.strip():
                counts[tag.strip()] += 1
    if not counts:
        return "no caption tags to report"
    shown = counts.most_common(top)
    lines = [f"{len(counts)} distinct tags across {len(files)} captioned images. Top {len(shown)}:"]
    width = max(len(tag) for tag, _ in shown)
    lines += [f"  {tag:<{width}}  {count}" for tag, count in shown]
    return "\n".join(lines)


# A comma part longer than this many words, not counting words in brackets, is a sentence.
_TAG_WORDS = 5
_IN_BRACKETS = re.compile(r"\\?\(.*?\)")  # also an escaped \(fate\)


def _is_sentence(part: str) -> bool:
    return len(_IN_BRACKETS.sub(" ", part).split()) > _TAG_WORDS


def tag_counts(image_dir: str | Path, caption_extension: str = ".txt") -> tuple[Counter[str], int, int]:
    """Count the caption files with each tag, all caption files, and the ones with sentences.

    long_hair and long hair are one tag, named as the captions spell it most often.
    """
    texts = []
    for caption in caption_files(image_dir, caption_extension):
        text = caption.read_text(encoding="utf-8", errors="replace")
        texts.append(_without_separator_lines(text, DEFAULT_MULTILINE_CAPTION_SEPARATOR))
    return _count_tags(texts)


def trained_captions(dataset_config: dict, train_config: dict) -> list[str]:
    """Each caption file as the trainer reads it: the first line, or with enable_multiline_caption the whole file."""
    captions = []
    for subset, scopes in subset_scopes(dataset_config, train_config):
        settings = _caption_settings(scopes)
        folder = Path(subset["image_dir"])
        if not folder.is_dir():
            continue
        separator = settings["multiline_caption_separator"] or DEFAULT_MULTILINE_CAPTION_SEPARATOR
        for image in image_files(folder):
            caption = _read_caption(image, settings)
            if caption is None:
                continue
            if settings["enable_multiline_caption"]:
                caption = _without_separator_lines(caption, separator)
            captions.append(caption)
    return captions


def _without_separator_lines(text: str, separator: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.strip().startswith(separator))


def normalize_tag(tag: str) -> str:
    """A tag as the tagger writes it with --remove_underscore: only tags longer than 3 characters change.

    See remove_underscore in finetune/tag_images_by_wd14_tagger.py.
    """
    tag = tag.strip()
    return tag.replace("_", " ") if len(tag) > 3 else tag


def _count_tags(texts: list[str]) -> tuple[Counter[str], int, int]:
    # tag_counts for caption texts.
    counts: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    sentences = 0
    for text in texts:
        parts = [part.strip() for line in text.splitlines() for part in line.split(",")]
        parts = [part for part in parts if part and not part.startswith("#")]
        if any(_is_sentence(part) for part in parts):
            sentences += 1
        # Once per file, in the caption's order, so ties keep a fixed order.
        tags = dict.fromkeys(part for part in parts if not _is_sentence(part))
        counts.update(dict.fromkeys(normalize_tag(tag) for tag in tags).keys())
        for tag in tags:
            spellings.setdefault(normalize_tag(tag), Counter())[tag] += 1
    # most_common keeps the first seen of equal counts.
    named = Counter({spellings[tag].most_common(1)[0][0]: count for tag, count in counts.items()})
    return named, len(texts), sentences


# Tags that say how many people are in the picture. They go first in a prompt.
_COUNT_TAG = re.compile(r"(\d+\+?(girl|boy|other)s?|solo|solo focus|no humans|multiple (girls|boys|others))")
# Tags about the picture, not the subject, in lowercase. The preview prompt sets its own framing.
_NOT_SUBJECT = frozenset(
    {
        "upper body",
        "cowboy shot",
        "full body",
        "lower body",
        "portrait",
        "close-up",
        "head out of frame",
        "feet out of frame",
        "out of frame",
        "looking at viewer",
        "looking to the side",
        "looking away",
        "looking back",
        "looking up",
        "looking down",
        "from side",
        "from behind",
        "from above",
        "from below",
        "straight-on",
        "profile",
        "dutch angle",
        "comic",
        "4koma",
        "2koma",
        "speech bubble",
        "english text",
        "text",
        "signature",
        "watermark",
        "artist name",
        "twitter username",
        "patreon username",
        "username",
        "web address",
        "dated",
        "copyright name",
        "character name",
        "multiple views",
        "reference sheet",
        "translated",
        "translation request",
        "commentary",
        "commentary request",
        "english commentary",
        "monochrome",
        "greyscale",
        "masterpiece",
        "best quality",
        "high quality",
        "good quality",
        "normal quality",
        "low quality",
        "worst quality",
        "absurdres",
        "highres",
        "lowres",
        "incredibly absurdres",
        "general",
        "safe",
        "sensitive",
        "questionable",
        "explicit",
        "nsfw",
        "very aesthetic",
        "aesthetic",
        "displeasing",
        "very displeasing",
        "bad quality",
        "great quality",
        "amazing quality",
        "high score",
        "great score",
        "good score",
        "average score",
        "bad score",
        "low score",
        "japanese text",
        "chinese text",
        "korean text",
        "sound effects",
        "logo",
        "blurry",
        "jpeg artifacts",
    }
)


def _tag_key(tag: str) -> str:
    return normalize_tag(tag).lower()


def _is_subject_tag(tag: str) -> bool:
    tag = _tag_key(tag)
    return not (
        tag in _NOT_SUBJECT
        or tag.endswith((" background", " text", " score", " quality"))
        or "aesthetic" in tag
        or tag.startswith(("score ", "rating", "year "))
        or tag in ("newest", "recent", "mid", "early", "old")
    )


def frequent_tags(dataset_config: dict, train_config: dict, min_share: float = 0.5, top: int = 10) -> list[str]:
    """The subject tags in at least ``min_share`` of the captions, for the preview prompt.

    Count tags such as 1girl come first. Empty when most captions are sentences.
    """
    counts, files, sentences = _count_tags(trained_captions(dataset_config, train_config))
    if not files or sentences * 2 > files:
        return []
    frequent = [tag for tag, count in counts.most_common() if count >= min_share * files and _is_subject_tag(tag)]
    people = [tag for tag in frequent if _COUNT_TAG.fullmatch(_tag_key(tag))]
    return (people + [tag for tag in frequent if tag not in people])[:top]


# The automatic num_repeats aims for this many images per epoch (each route sets
# its own repeats_target), and never goes above MAX_AUTO_REPEATS.
AUTO_REPEATS_TARGET = 400
MAX_AUTO_REPEATS = 20


def default_num_repeats(image_count: int, target_per_epoch: int = AUTO_REPEATS_TARGET) -> int:
    """A repeat count for small datasets that aims for about ``target_per_epoch`` images per epoch."""
    if image_count <= 0:
        return 1
    return max(1, min(MAX_AUTO_REPEATS, round(target_per_epoch / image_count)))


def build_dataset_config(
    image_dir: str | Path,
    *,
    resolution: int = 1024,
    batch_size: int = 2,
    num_repeats: int = 1,
    caption_extension: str = ".txt",
    class_tokens: str | None = None,
    caption_prefix: str | None = None,
    shuffle_caption: bool = False,
    keep_tokens: int = 0,
    enable_bucket: bool = True,
    min_bucket_reso: int = 256,
    max_bucket_reso: int = 2048,
    bucket_reso_steps: int = 64,
    bucket_no_upscale: bool = False,
    flip_aug: bool = False,
    folder_class_tokens: bool = True,
) -> dict:
    """Build the [general]/[[datasets]]/[[datasets.subsets]] structure as a dict.

    A ``<N>_<name>`` folder's N wins over ``num_repeats`` and its name over
    ``class_tokens``. ``folder_class_tokens=False`` leaves the folder names out of the captions.
    """
    general: dict[str, object] = {
        "resolution": resolution,
        "enable_bucket": enable_bucket,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
        "bucket_reso_steps": bucket_reso_steps,
        "bucket_no_upscale": bucket_no_upscale,
        "caption_extension": caption_extension,
        "shuffle_caption": shuffle_caption,
        "keep_tokens": keep_tokens,
        "flip_aug": flip_aug,
    }
    subsets = []
    for found in find_subsets(image_dir).subsets:
        subset: dict[str, object] = {
            "image_dir": str(Path(found.image_dir).resolve()),
            "num_repeats": found.num_repeats or num_repeats,
        }
        tokens = (found.name if folder_class_tokens else "") or class_tokens
        if tokens:
            subset["class_tokens"] = tokens
        if caption_prefix:
            subset["caption_prefix"] = caption_prefix
        _assert_known(subset, SUBSET_KEYS, "[[datasets.subsets]]")
        subsets.append(subset)

    _assert_known(general, GENERAL_KEYS, "[general]")
    return {"general": general, "datasets": [{"batch_size": batch_size, "subsets": subsets}]}


def _assert_known(mapping: dict, allowed: frozenset, where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"keys not in the sd-scripts dataset schema for {where}: {unknown}")


def refresh_text_caches(dataset_config: dict, train_config: dict, entrypoint: str) -> None:
    """Remove a folder's old text-encoder cache files when what they encode changed.

    The backend reuses a cache file without checking the text encoder, or on
    SDXL and Anima the caption. Each folder keeps a record of what its cache
    was made from. Latent caches stay.
    """
    suffix = TEXT_CACHE_SUFFIXES.get(entrypoint)
    if suffix is None or not train_config.get("cache_text_encoder_outputs_to_disk"):
        return
    encoder = _digest({key: _arg_identity(key, train_config.get(key)) for key in TEXT_CACHE_ARGS[entrypoint]})
    folders: dict[Path, list] = {}
    for subset, scopes in subset_scopes(dataset_config, train_config):
        settings = _caption_settings(scopes)
        folder = Path(subset["image_dir"])
        if not folder.is_dir():
            continue  # the backend reports a missing folder itself
        captions = _cached_captions(folder, settings)
        # The captions already hold these.
        kept = {key: value for key, value in settings.items() if key not in _JOINED_SETTINGS}
        folders.setdefault(folder, []).append({"settings": kept, "captions": captions})
    for folder, subsets in folders.items():
        current = {"captions": _digest(subsets), "encoder": encoder}
        record_path = folder / TEXT_CACHE_RECORD
        record = _read_record(record_path)
        previous = record.get(suffix)
        if previous == current:
            continue
        removed = _remove_text_caches(folder, suffix)
        if removed:
            print(
                f"removed {removed} old text-encoder cache file{'s' if removed != 1 else ''} from {folder}: "
                f"{_stale_reason(previous, current)}. Training makes new ones."
            )
        record[suffix] = current
        try:
            record_path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
        except OSError:
            pass  # the next run removes the caches again: slower, never wrong


def captions_with_candidates(dataset_config: dict, train_config: dict) -> list[Path]:
    """The images whose caption enable_multiline_caption splits into more than one candidate.

    See split_caption_candidates in library/dataset.py.
    """
    return [
        image
        for image, caption, separator in _multiline_captions(dataset_config, train_config)
        if _candidate_count(caption, separator) > 1
    ]


def bad_separator_lines(dataset_config: dict, train_config: dict) -> list[tuple[Path, str]]:
    """Each image whose caption has a separator line with text after it, which the backend stops on."""
    found = []
    for image, caption, separator in _multiline_captions(dataset_config, train_config):
        for line in caption.split("\n"):
            stripped = line.strip()
            if stripped.startswith(separator) and stripped[len(separator) :].split("#", 1)[0].strip():
                found.append((image, stripped))
                break
    return found


def _multiline_captions(dataset_config: dict, train_config: dict):
    # (image, caption, separator) for each caption read with enable_multiline_caption.
    for subset, scopes in subset_scopes(dataset_config, train_config):
        settings = _caption_settings(scopes)
        folder = Path(subset["image_dir"])
        if not settings["enable_multiline_caption"] or not folder.is_dir():
            continue
        separator = settings["multiline_caption_separator"] or DEFAULT_MULTILINE_CAPTION_SEPARATOR
        for image in image_files(folder):
            caption = _read_caption(image, settings)
            if caption is not None:
                yield image, caption, separator


def _candidate_count(caption: str, separator: str) -> int:
    # Mirrors split_caption_candidates in library/dataset.py.
    parts: list[list[str]] = [[]]
    for line in caption.split("\n"):
        if line.strip().startswith(separator):
            parts.append([])
        else:
            parts[-1].append(line)
    return sum(1 for part in parts if "\n".join(part).strip())


def subset_scopes(dataset_config: dict, train_config: dict):
    """Each subset of a dataset config, with the places the backend looks up its settings, in order.

    The order is the backend's (config_util.py).
    """
    general = dataset_config.get("general", {})
    for ds in dataset_config.get("datasets", []):
        for subset in ds.get("subsets", []):
            yield subset, (subset, ds, general, train_config)


def scoped_value(key: str, scopes: tuple, default=None):
    """The first value of ``key`` in ``scopes`` that is not None, or ``default``."""
    for scope in scopes:
        value = scope.get(key)
        if value is not None:
            return value
    return default


def _caption_settings(scopes: tuple) -> dict:
    return {key: scoped_value(key, scopes, default) for key, default in _CAPTION_SETTINGS.items()}


def _read_caption(image: Path, settings: dict) -> str | None:
    # Mirrors read_caption in library/dreambooth_dataset.py, which also tries the stem without its
    # last four "_" parts.
    base = os.path.splitext(str(image))[0]
    tokens = base.split("_")
    candidates = [base, "_".join(tokens[:-4]) if len(tokens) >= 5 else base]
    for candidate in candidates:
        path = Path(candidate + settings["caption_extension"])
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if settings["enable_multiline_caption"]:
                return text
            if settings["enable_wildcard"]:
                return "\n".join(line.strip() for line in text.split("\n") if line.strip())
            return text.split("\n")[0].strip()
    return None


def _cached_captions(folder: Path, settings: dict) -> list:
    """Each image's caption as the backend encodes it for the cache."""
    captions = []
    # Any case: an extra image only costs a rebuild.
    images = (p for p in folder.iterdir() if not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS)
    for image in sorted(images):
        caption = _read_caption(image, settings)
        if caption is None:
            caption = settings["class_tokens"] or ""
        # Joined as in add_caption_prefix_suffix (library/dataset.py).
        if settings["caption_prefix"]:
            caption = settings["caption_prefix"] + " " + caption
        if settings["caption_suffix"]:
            caption = caption + " " + settings["caption_suffix"]
        captions.append([image.name, caption])
    return captions


def _arg_identity(key: str, value):
    """A model file by name, size and a sample of its bytes, so its folder never matters."""
    if key not in _MODEL_FILE_ARGS or value is None:
        return value
    path = Path(str(value))
    if not path.is_file():
        return path.name
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(size // 2)
        sample = hashlib.sha256(handle.read(1 << 20)).hexdigest()
    return [path.name, size, sample]


def _digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _remove_text_caches(folder: Path, suffix: str) -> int:
    # Never a folder. A file link is removed, never its target.
    def matches(name: str) -> bool:
        return (name.lower() if os.name == "nt" else name).endswith(suffix)

    with os.scandir(folder) as entries:
        stale = [entry.path for entry in entries if matches(entry.name) and not entry.is_dir()]
    try:
        for path in stale:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue  # already gone, for example removed by a Drive sync
    except OSError as error:
        raise RuntimeError(
            f"could not remove the old text-encoder cache files in {folder} ({error.strerror}). "
            "Make the folder writable, or keep the cache in memory only by adding "
            "'cache_text_encoder_outputs_to_disk': None to config_overrides"
        ) from error
    return len(stale)


def _stale_reason(previous, current: dict) -> str:
    if not isinstance(previous, dict):
        return "there is no record of what they were made from"
    changed = [part for part in ("captions", "encoder") if previous.get(part) != current[part]]
    if changed == ["captions"]:
        return "the captions changed since they were made"
    if changed == ["encoder"]:
        return "the text encoder or its settings changed since they were made"
    return "the captions and the text encoder changed since they were made"

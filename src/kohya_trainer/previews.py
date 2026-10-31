"""Sample previews: the preview size, the built-in prompt and the prompt file."""

from __future__ import annotations

import re
from pathlib import Path

from . import config as config_mod
from . import dataset as dataset_mod
from . import tagging
from .registry import ModelRoute


def _image_size(value, fallback: int) -> tuple[int, int]:
    """(width, height) from a dataset resolution: one number, or a width and height pair."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value[0], value[1]
    return fallback, fallback


def sample_size(width: int, height: int) -> tuple[int, int]:
    """The preview size for a training size: a square one becomes portrait.

    Above 640, a square r becomes 0.8125r x 1.1875r, both rounded to 64 (832x1216
    at 1024). At 640 and below, the width stays r and the height is 1.5r, rounded
    to 64 (512x768 at 512).
    """
    if width != height:
        return width, height

    def to_64(value: float) -> int:
        return max(64, round(value / 64) * 64)

    if width > 640:
        return to_64(width * 0.8125), to_64(width * 1.1875)
    return width, to_64(width * 1.5)


# At portrait sizes, upper body can cut off the top of the head.
SAMPLE_FRAMING = "cowboy shot"
SAMPLE_SCENE = "looking at viewer, simple background"
# A fixed seed, so the previews of each epoch can be compared.
SAMPLE_SEED = 42
# These trainers read (x) and [x] as prompt weights, so brackets in tags are escaped.
_WEIGHTED_PROMPTS = frozenset({"train_network.py", "sdxl_train_network.py"})
_BRACKET = re.compile(r"(?<!\\)([()\[\]])")


def default_sample_prompt(route: ModelRoute, trigger_word: str = "", subject_tags: list[str] | None = None) -> str:
    """The built-in preview prompt: the route's quality tags, the trigger, the dataset's common tags and a plain scene.

    The trigger and the tags in the route's negative prompt are left out of the
    common tags. Without common tags the subject is ``solo`` (``alone`` in a
    sentence). A route that learned from sentences gets a sentence.
    """
    trigger = trigger_word.strip()
    own = tagging.normalize_tag(trigger)
    negative = {tagging.normalize_tag(tag).lower() for tag in route.default_sample_negative.split(",")}
    tags = [
        tag
        for tag in subject_tags or []
        if tagging.normalize_tag(tag) != own and tagging.normalize_tag(tag).lower() not in negative
    ]
    if route.sample_sentence:
        lead = [trigger] if trigger else [] if tags else ["A character"]
        subject = ", ".join([*lead, *(tags or ["alone"])])
        return (
            f"{subject}, looking at the viewer. A {SAMPLE_FRAMING} against a simple background, {route.sample_quality}."
        )
    subject = tags or ["solo"]
    if route.entrypoint in _WEIGHTED_PROMPTS:
        subject = [_BRACKET.sub(r"\\\1", tag) for tag in subject]
    parts = [trigger] if trigger else []
    parts += [*subject, SAMPLE_FRAMING, SAMPLE_SCENE]
    if route.sample_quality_first:
        parts.insert(0, route.sample_quality)
    else:
        parts.append(route.sample_quality)
    return ", ".join(parts)


def sample_prompt_lines(sample_prompt: str | None, default_prompt: str) -> list[str]:
    """The sample prompts, from a prompt, a .txt file with one per line, or ``default_prompt``."""
    text = (sample_prompt or "").strip()
    if not text:
        return [default_prompt]
    if text.lower().endswith(".txt"):
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(
                f"sample_prompt names a .txt file that does not exist: {text}. Give the path of a file "
                "with one prompt per line, or type the prompt itself."
            )
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        if not [line for line in lines if not line.startswith("#")]:
            raise ValueError(f"the sample prompt file {text} has no prompts in it")
        return lines
    return [text]


def _with_prompt_defaults(line: str, defaults: dict) -> str:
    """Add each default flag the line does not set. "--ss" (the sampler) is not "--s" (library/sampling.py)."""
    if line.startswith("#"):
        return line
    given = {part.split(" ", 1)[0].lower() for part in line.split(" --")[1:]}
    extra = "".join(
        f" --{flag} {value}" for flag, value in defaults.items() if flag not in given and str(value).strip()
    )
    return line + extra


def _write_prompts(
    route: ModelRoute,
    train_config: dict,
    dataset_config: dict,
    path: Path,
    trigger_word: str,
    sample_prompt: str | None,
    sample_negative_prompt: str | None,
    size: tuple[int, int],
) -> Path:
    """Write the preview prompts with this route's sample settings, and return the file.

    A prompt the user wrote gets only the flags it lacks. A negative of None
    takes the route's, and "" means none.
    """
    negative = route.default_sample_negative if sample_negative_prompt is None else sample_negative_prompt.strip()
    sample_width, sample_height = sample_size(*size)
    defaults = {
        "n": negative,
        "w": sample_width,
        "h": sample_height,
        "s": route.sample_steps,
        "l": f"{route.sample_cfg:g}",
        "d": SAMPLE_SEED,
    }
    built_in = not (sample_prompt or "").strip()
    subject_tags = dataset_mod.frequent_tags(dataset_config, train_config) if built_in else []
    default_prompt = default_sample_prompt(route, trigger_word, subject_tags)
    if built_in:
        print(f"preview prompt ({sample_width}x{sample_height}): {default_prompt}")
    prompts = [_with_prompt_defaults(line, defaults) for line in sample_prompt_lines(sample_prompt, default_prompt)]
    return config_mod.write_sample_prompts(prompts, path)

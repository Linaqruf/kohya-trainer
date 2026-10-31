"""Build the sd-scripts trainer config (--config_file TOML).

sd-scripts reads the TOML as one flat list of arguments (library/args.py), so the sectioned
presets are flattened here. Every key is checked against the pinned backend,
which ignores an unknown key without a word.
"""

from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import tomli_w

from . import dataset as dataset_mod
from .registry import BaseModel, ModelRoute

# Flags for a v-prediction base such as NoobAI v-pred, as the Illustrious
# community trains them. min_snr_gamma and noise_offset are dropped with them.
VPRED_FLAGS = {
    "v_parameterization": True,
    "zero_terminal_snr": True,
    "scale_v_pred_loss_like_noise_pred": True,
    "debiased_estimation_loss": True,
    # The NoobAI v-pred card: "Euler (Other samplers will not work properly)".
    "sample_sampler": "euler",
}

_ADD_ARGUMENT = re.compile(r"""add_argument\(\s*['"]--([A-Za-z0-9_]+)['"]""")

# Trainers whose text-encoder cache still allows caption dropout (their trainers pass
# cache_supports_dropout=True). A test checks the pin.
TE_CACHE_KEEPS_DROPOUT = frozenset({"anima_train_network.py", "moss_image_train_network.py"})

# Trainers whose text-encoder cache keeps every caption candidate of an image
# (enable_multiline_caption). moss_image_train_network.py passes
# supports_multiple_caption_candidates=True. A test checks the pin.
TE_CACHE_KEEPS_CANDIDATES = frozenset({"moss_image_train_network.py"})

# A library/ file-name prefix, and the only entrypoint that loads those arguments.
_ENTRYPOINT_ONLY_LIBRARIES = {"moss_image_": "moss_image_train_network.py"}


def load_preset(name: str) -> dict:
    """Load a bundled preset and flatten its sections, as sd-scripts does."""
    text = (resources.files("kohya_trainer") / "presets" / name).read_text(encoding="utf-8")
    return flatten(tomllib.loads(text))


def flatten(data: dict) -> dict:
    """Flatten nested TOML tables to one level of argparse dest names. A duplicate key raises."""
    flat: dict[str, object] = {}

    def _walk(mapping: dict) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                _walk(value)
            else:
                if key in flat:
                    raise ValueError(f"duplicate key {key!r} while flattening config")
                flat[key] = value

    _walk(data)
    return flat


def known_args(sd_scripts_dir: str | Path, entrypoint: str) -> frozenset[str]:
    """The argument names the backend accepts, read from its source.

    Reads the entrypoint, train_network.py and everything under library/.
    It accepts more than one script takes, to catch typos only. Library code
    that only one entrypoint loads counts only for that entrypoint.
    """
    root = Path(sd_scripts_dir)
    sources = [root / entrypoint, root / "train_network.py"]
    for source in sorted((root / "library").glob("*.py")):
        owner = next((e for prefix, e in _ENTRYPOINT_ONLY_LIBRARIES.items() if source.name.startswith(prefix)), None)
        if owner is None or owner == Path(entrypoint).name:
            sources.append(source)
    names: set[str] = set()
    for source in sources:
        if source.is_file():
            names.update(_ADD_ARGUMENT.findall(source.read_text(encoding="utf-8", errors="replace")))
    if not names:
        raise FileNotFoundError(f"no argparse definitions found under {root}. Is the submodule checked out?")
    return frozenset(names)


def validate_keys(config: dict, sd_scripts_dir: str | Path, entrypoint: str) -> None:
    """Raise ValueError for any config key the backend does not accept."""
    unknown = sorted(set(config) - known_args(sd_scripts_dir, entrypoint))
    if unknown:
        raise ValueError(
            f"config keys not accepted by {entrypoint} at the pinned sd-scripts commit: {unknown}. "
            "sd-scripts ignores them without a word."
        )


def build_train_config(
    route: ModelRoute,
    base: BaseModel,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    output_name: str,
    vae_path: str | Path | None = None,
    sample_prompts: str | Path | None = None,
    overrides: dict | None = None,
    sd_scripts_dir: str | Path | None = None,
) -> dict:
    """Build the flat trainer config from the preset, the run's values and the base's flags.

    ``overrides`` wins, and None removes the key. ``sd_scripts_dir`` turns on the key check.
    """
    config = load_preset(route.preset)
    config["pretrained_model_name_or_path"] = str(checkpoint_path)
    config["output_dir"] = str(output_dir)
    config["output_name"] = output_name
    if vae_path:
        config["vae"] = str(vae_path)
    if sample_prompts:
        config["sample_prompts"] = str(sample_prompts)
    if base.v_prediction:
        config.update(VPRED_FLAGS)
    for flag in base.convrot_flags:
        config[flag] = True  # the backend quantizes this run's bf16 file while it loads it
    for key, value in (overrides or {}).items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    if config.get("debiased_estimation_loss") and "min_snr_gamma" in config:
        # The two SNR weightings never both apply. For min-SNR on a v-pred
        # base, set debiased_estimation_loss to None.
        config.pop("min_snr_gamma")
    if config.get("zero_terminal_snr") and "noise_offset" in config:
        # On a ztSNR base, offset noise only skews the trained range.
        config.pop("noise_offset")
    if sd_scripts_dir is not None:
        validate_keys(config, sd_scripts_dir, route.entrypoint)
    return config


def check_compat(train_config: dict, dataset_config: dict, entrypoint: str = "") -> None:
    """Refuse caption options that a text-encoder cache cannot apply, before the trainer does."""
    if not train_config.get("cache_text_encoder_outputs") and not train_config.get(
        "cache_text_encoder_outputs_to_disk"
    ):
        return
    keys = ["shuffle_caption", "caption_tag_dropout_rate", "token_warmup_step"]
    if entrypoint not in TE_CACHE_KEEPS_DROPOUT:
        keys.append("caption_dropout_rate")
    offenders = set()
    for _, scopes in dataset_mod.subset_scopes(dataset_config, train_config):
        offenders.update(key for key in keys if dataset_mod.scoped_value(key, scopes))
    if offenders:
        raise ValueError(
            f"text-encoder caching cannot be used with {sorted(offenders)}: the cache holds one fixed caption "
            "per image. Drop those options, or turn caching off by adding "
            "'cache_text_encoder_outputs': None, 'cache_text_encoder_outputs_to_disk': None to config_overrides "
            "(slower, and it uses more GPU memory)"
        )
    check_text_encoder_training(train_config)


def check_text_encoder_training(train_config: dict) -> None:
    """Refuse a text-encoder LoRA with cached text-encoder outputs.

    The trainers refuse the pair in assert_extra_args, after the dataset loads
    and before the models load.
    The fix also names 'network_train_unet_only': False. Without it, a user who
    copies only the caching keys gets the preset's unet-only back, without a word.
    """
    caches = train_config.get("cache_text_encoder_outputs") or train_config.get("cache_text_encoder_outputs_to_disk")
    if caches and not train_config.get("network_train_unet_only"):
        raise ValueError(
            "cache_text_encoder_outputs requires network_train_unet_only. To train the text encoder, "
            "turn the caching off as well and give the text encoder a learning rate. Add "
            "'cache_text_encoder_outputs': None, 'cache_text_encoder_outputs_to_disk': None, "
            "'network_train_unet_only': False, 'text_encoder_lr': 1e-4 to config_overrides (needs more GPU memory)"
        )


def write_toml(config: dict, path: str | Path) -> Path:
    """Write ``config`` as TOML to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config), encoding="utf-8", newline="\n")
    return path


def write_sample_prompts(prompts: str | list[str], path: str | Path) -> Path:
    """Write a sample-prompts .txt (one prompt per line, sd-scripts --opt syntax)."""
    if isinstance(prompts, str):
        prompts = [prompts]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8", newline="\n")
    return path

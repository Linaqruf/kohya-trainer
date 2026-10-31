"""Optimizers and lr schedulers: the names offered, the checks, and a run's optimizer settings."""

from __future__ import annotations

import ast
import difflib
import functools
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

# The optimizers the notebooks offer. Each has its package in the backend
# requirements. Any other backend name works if its package is installed.
OPTIMIZERS = ("AdamW8bit", "AdamW", "Prodigy", "ProdigyScheduleFree", "CAME", "AdaFactor")
# The backend loads a dotted name as module.Class (library/optimizer.py).
OPTIMIZER_NAMES = {
    "CAME": "pytorch_optimizer.CAME",
    "ProdigyScheduleFree": "prodigyplus.ProdigyPlusScheduleFree",
}
# The recommended settings (optimizer_recommended). They replace only form values.
OPTIMIZER_DEFAULTS: dict[str, dict] = {
    # The form's scheduler runs. cosine or constant_with_warmup are usual with Prodigy.
    "Prodigy": {
        "learning_rate": 1.0,
        "unet_lr": 1.0,
        "text_encoder_lr": 1.0,
        "optimizer_args": (
            "decouple=True",
            "weight_decay=0.01",
            "d_coef=2",
            "use_bias_correction=True",
            "safeguard_warmup=True",
        ),
    },
    # Its package says not to clip gradients (prodigy-plus-schedule-free README). The presets set
    # max_grad_norm 1.0.
    "ProdigyScheduleFree": {"learning_rate": 1.0, "unet_lr": 1.0, "text_encoder_lr": 1.0, "max_grad_norm": 0.0},
    # CAME keeps the form's learning rate. 1e-4 to 2e-4 is its usual range.
    "CAME": {"optimizer_args": ("weight_decay=0.04",)},
    "AdaFactor": {
        "optimizer_args": ("scale_parameter=False", "relative_step=False", "warmup_init=False"),
    },
}

# The backend falls back to torch.optim. This list is used only if torch cannot be imported.
TORCH_OPTIMIZERS = (
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "LBFGS",
    "Muon",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
)

# The lr_scheduler names the pinned backend can build (get_scheduler_fix in
# library/optimizer.py). The canary checks these against the installed transformers.
TRANSFORMERS_SCHEDULERS = (
    "constant",
    "constant_with_warmup",
    "cosine",
    "cosine_with_restarts",
    "linear",
    "polynomial",
    "inverse_sqrt",
    "warmup_stable_decay",
    "cosine_with_min_lr",
)
SCHEDULERS = (*TRANSFORMERS_SCHEDULERS, "piecewise_constant")
# Names the backend knows but fails on, and why.
_CANT_BUILD = "the backend passes it settings it does not take, so it fails when training starts"
REFUSED_SCHEDULERS = {
    "reduce_lr_on_plateau": _CANT_BUILD,
    "greedy": _CANT_BUILD,
    "cosine_warmup_with_min_lr": _CANT_BUILD,
    "adafactor": (
        "the backend sets that scheduler itself when AdaFactor runs in relative-step mode, "
        "and refuses it with any other optimizer"
    ),
}

_BACKEND_OPTIMIZER = re.compile(r"""optimizer_type\s*==\s*['"]([A-Za-z0-9_]+)['"]\.lower\(\)""")


def _friendly_optimizer(name: str) -> str | None:
    """The OPTIMIZERS entry that ``name`` runs, or None. Only plain names match without case."""
    for friendly in OPTIMIZERS:
        backend = OPTIMIZER_NAMES.get(friendly, friendly)
        if name == backend or name.lower() == friendly.lower():
            return friendly
        if "." not in backend and name.lower() == backend.lower():
            return friendly
    return None


@functools.cache
def backend_optimizer_names(sd_scripts_dir: str) -> frozenset[str]:
    """The plain optimizer names the backend handles itself, read from library/optimizer.py."""
    source = Path(sd_scripts_dir) / "library" / "optimizer.py"
    names = _BACKEND_OPTIMIZER.findall(source.read_text(encoding="utf-8", errors="replace"))
    if not names:
        raise FileNotFoundError(f"no optimizer names found in {source}. Is the submodule checked out?")
    return frozenset(names)


def _torch_optimizer_names() -> tuple[str, ...]:
    try:
        import torch.optim
    except ImportError:
        return TORCH_OPTIMIZERS
    return tuple(name for name in dir(torch.optim) if isinstance(getattr(torch.optim, name), type))


# The package each family of backend optimizer names imports (library/optimizer.py). None means
# torch or transformers.
def _backend_package(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("dadapt"):
        return "dadaptation"
    if lower == "prodigy":
        return "prodigyopt"
    if lower == "lion":
        return "lion_pytorch"
    if lower.endswith("8bit") or lower.startswith("paged"):
        return "bitsandbytes"
    if lower.endswith("schedulefree"):
        return "schedulefree"
    return None


# An import name whose package has another name in requirements.txt.
_IMPORT_PACKAGES = {"prodigyplus": "prodigy_plus_schedule_free"}
# The setup cell installs torch by its own rule (pins.py), not from requirements.txt.
_SETUP_PACKAGES = frozenset({"torch", "torchvision"})


@functools.cache
def pinned_packages(sd_scripts_dir: str) -> frozenset[str]:
    """The packages the setup cell installs, in lower case with "_" for "-"."""
    source = Path(sd_scripts_dir) / "requirements.txt"
    names = set(_SETUP_PACKAGES)
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#")[0].strip()
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match and not line.startswith("-"):
            names.add(match.group(0).lower().replace("-", "_"))
    return frozenset(names)


def _package_installed(package: str, sd_scripts_dir: str | Path) -> bool:
    """True when the setup cell installs ``package``, or it is installed here."""
    pinned = _IMPORT_PACKAGES.get(package, package) in pinned_packages(str(sd_scripts_dir))
    return pinned or importlib.util.find_spec(package) is not None


def check_optimizer(name: str, sd_scripts_dir: str | Path) -> None:
    """Refuse an optimizer name that makes the backend fail, before any caching.

    A dotted name whose package is not installed here, as in CI, must be in
    the backend's requirements.
    """
    if _friendly_optimizer(name):
        return
    if "." in name:
        module, _, attribute = name.rpartition(".")
        top = module.partition(".")[0]
        if importlib.util.find_spec(top) is None:
            if not _package_installed(top, sd_scripts_dir):
                raise ValueError(
                    f"optimizer {name!r}: no package {top!r} is installed, and the backend's requirements do not "
                    "install one, so the backend cannot load it. Check the spelling, or install its package first."
                )
            return
        try:
            loaded = importlib.import_module(module)
        except ImportError as error:
            raise ValueError(
                f"optimizer {name!r}: {module!r} cannot be imported ({error}). Check the spelling."
            ) from error
        if not hasattr(loaded, attribute):
            public = [entry for entry in dir(loaded) if not entry.startswith("_")]
            close = difflib.get_close_matches(attribute, public, n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(f'{module}.{c}' for c in close)}?" if close else ""
            raise ValueError(f"optimizer {name!r}: {module} has no {attribute!r}.{hint} Check the spelling.")
        return
    backend = backend_optimizer_names(str(sd_scripts_dir))
    torch_names = _torch_optimizer_names()
    if name.lower() in {known.lower() for known in backend}:
        package = _backend_package(name)
        if package and not _package_installed(package, sd_scripts_dir):
            raise ValueError(
                f"optimizer {name!r} needs the {package} package, which the backend's requirements do not install, "
                f"so it fails when training starts. Pick another optimizer, or install {package} first."
            )
        return
    if name in torch_names:
        return
    usable = [k for k in backend if not _backend_package(k) or _package_installed(_backend_package(k), sd_scripts_dir)]
    known = sorted({*OPTIMIZERS, *usable, *torch_names})
    close = difflib.get_close_matches(name.lower(), [k.lower() for k in known], n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(next(k for k in known if k.lower() == c) for c in close)}?" if close else ""
    raise ValueError(
        f"optimizer {name!r} is not one the backend knows.{hint} An optimizer from an installed package can be "
        "written as package.Name, for example pytorch_optimizer.CAME."
    )


_SCHEDULER_CLASS_HINT = (
    'To use a scheduler class from a package, set "lr_scheduler_type" in config_overrides instead. It '
    'then runs with no warmup unless "lr_warmup_steps" is in config_overrides too.'
)


def adafactor_relative_step(optimizer_args) -> bool:
    """True when AdaFactor runs in relative-step mode, where the backend sets its own schedule.

    warmup_init turns the mode on even with relative_step=False.
    """
    values = optimizer_arg_values(optimizer_args)
    return bool(values.get("relative_step", True)) or bool(values.get("warmup_init", False))


def optimizer_arg_values(optimizer_args) -> dict:
    """Read optimizer_args as the backend does (ast.literal_eval in library/optimizer.py).

    Refuse what it cannot read.
    """
    values = {}
    for entry in (str(arg) for arg in optimizer_args or []):
        if entry.count("=") != 1:
            raise ValueError(f"optimizer_args entry {entry!r} needs exactly one '='. Write each setting as name=value.")
        name, _, raw = entry.partition("=")
        try:
            values[name.strip()] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"optimizer_args entry {entry!r}: the backend cannot read {raw!r} as a value. Use True or False "
                "(with a capital letter), a number, or quotes around text."
            ) from None
    return values


def check_scheduler(name: str, settings: dict | None = None) -> None:
    """Refuse an lr_scheduler that makes the backend fail, before any caching.

    ``settings`` gives cosine_with_min_lr its floor and piecewise_constant its step rules.
    """
    settings = settings or {}
    refused = REFUSED_SCHEDULERS.get("adafactor" if name.startswith("adafactor") else name)
    if refused:
        raise ValueError(f"lr_scheduler {name!r} cannot be used here: {refused}. Pick another scheduler.")
    if name not in SCHEDULERS:
        close = difflib.get_close_matches(name, SCHEDULERS, n=3, cutoff=0.6)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"lr_scheduler {name!r} is not a scheduler the backend knows.{hint} {_SCHEDULER_CLASS_HINT}")
    args = settings.get("lr_scheduler_args") or []
    if isinstance(args, str):
        args = args.split()
    arg_names = {str(arg).partition("=")[0].strip() for arg in args}
    if name == "cosine_with_min_lr":
        # The backend passes the ratio as min_lr_rate (get_scheduler_fix). transformers takes one
        # floor, min_lr or min_lr_rate, and refuses both or neither.
        ratio = settings.get("lr_scheduler_min_lr_ratio") is not None
        if "min_lr_rate" in arg_names:
            raise ValueError(
                'lr_scheduler cosine_with_min_lr: put the floor in "lr_scheduler_min_lr_ratio", not as min_lr_rate '
                "in lr_scheduler_args. The backend passes that ratio as min_lr_rate itself."
            )
        if ratio and "min_lr" in arg_names:
            raise ValueError(
                'lr_scheduler cosine_with_min_lr takes one floor: "lr_scheduler_min_lr_ratio" or min_lr in '
                "lr_scheduler_args, not both."
            )
        if not ratio and "min_lr" not in arg_names:
            raise ValueError(
                "lr_scheduler cosine_with_min_lr needs the lowest learning rate it ends at. Add "
                '"lr_scheduler_min_lr_ratio" to config_overrides, such as 0.1 for a tenth of the learning rate '
                '(or "lr_scheduler_args": ["min_lr=1e-6"] for a fixed rate).'
            )
    if name == "piecewise_constant" and "step_rules" not in arg_names:
        raise ValueError(
            'lr_scheduler piecewise_constant needs its steps. Add "lr_scheduler_args": '
            "[\"step_rules='1:500,0.1'\"] to config_overrides (the full rate for 500 steps, then a tenth of it)."
        )


def parse_optimizer_args(text: str) -> list[str]:
    """The optimizer_args field as a list of name=value entries.

    A piece without "=" joins the entry before it, so "betas=0.9, 0.999" stays one entry.
    """
    entries: list[str] = []
    for piece in (text or "").split():
        if "=" in piece or not entries:
            entries.append(piece)
        else:
            entries[-1] += piece
    entries = [entry.rstrip(",") for entry in entries]
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"optimizer_args entry {entry!r} has no '='. Write each setting as name=value.")
    return entries


def merge_args(*groups) -> list[str]:
    """Join name=value lists. A later list's entry replaces an earlier one with the same name."""
    merged: dict[str, str] = {}
    for group in groups:
        for entry in group or ():
            merged[str(entry).partition("=")[0].strip()] = str(entry)
    return list(merged.values())


def preset_optimizer_args(preset: dict, optimizer_type: str) -> list[str]:
    """The preset's optimizer_args when the preset's own optimizer runs, else none.

    Anima's betas=0.9,0.99 break CAME, whose betas take three values.
    """
    runs = str(optimizer_type or preset.get("optimizer_type", ""))
    own = str(preset.get("optimizer_type", ""))
    same = runs == own if "." in runs or "." in own else runs.lower() == own.lower()
    return [str(arg) for arg in preset.get("optimizer_args") or ()] if same else []


def optimizer_that_runs(optimizer: str, config_overrides: dict, preset: dict) -> str:
    """The optimizer name the run uses: one in config_overrides wins over the dropdown."""
    if "optimizer_type" in config_overrides:
        return str(config_overrides["optimizer_type"] or "")
    return str(optimizer or preset.get("optimizer_type") or "")


@dataclass(frozen=True)
class _RunSettings:
    """The settings that build_plan and check_before_downloads both work out."""

    optimizer: str  # as written, or "" for the backend's default
    friendly: str | None  # its OPTIMIZERS entry
    optimizer_type: str  # the optimizer_type the config gets
    recommended: dict  # the recommended settings that apply
    replaced: list[str]  # recommended keys that replace the form's values
    follows_lr: list[str]  # keys that take the learning_rate the user typed
    optimizer_args: list[str] | None  # None when config_overrides removes them all
    scheduler: str  # the lr_scheduler name
    # Why the backend does not build lr_scheduler: "schedule-free", "class" or "adafactor".
    scheduler_unused: str
    zero_te_rate: bool  # the text-encoder rate is 0, but the text encoder still trains


def _learning_rates(values: dict, user: dict, recommended: dict) -> tuple[list[str], list[str], dict]:
    """Return the keys that follow the user's learning_rate, the recommended keys that apply, and the rates."""
    # The backend lets unet_lr and text_encoder_lr win over learning_rate.
    # A text_encoder_lr from the form is the user's own field and stays.
    user_lr = user.get("learning_rate")
    follows_lr: list[str] = []
    if user_lr is not None:
        if "unet_lr" not in user and (values.get("unet_lr") is not None or "unet_lr" in recommended):
            follows_lr.append("unet_lr")
        if "text_encoder_lr" not in user and "text_encoder_lr" in recommended:
            follows_lr.append("text_encoder_lr")
    replaced = [key for key in recommended if key != "optimizer_args" and key not in user and key not in follows_lr]
    rates = {**values, **{key: recommended[key] for key in replaced}, **dict.fromkeys(follows_lr, user_lr)}
    return follows_lr, replaced, rates


def _merged_optimizer_args(
    preset: dict, user: dict, recommended: dict, optimizer_type: str, optimizer_args: str
) -> list[str] | None:
    """The merged optimizer_args, or None when config_overrides removes them all."""
    user_args = user.get("optimizer_args", ())
    if user_args is None:
        return None
    if isinstance(user_args, str):
        user_args = parse_optimizer_args(user_args)
    # Each wins over the one before it, entry by entry.
    return merge_args(
        preset_optimizer_args(preset, optimizer_type),
        recommended.get("optimizer_args"),
        parse_optimizer_args(optimizer_args),
        user_args,
    )


def _scheduler_unused(optimizer_type: str, form: dict, user: dict, merged_args: list[str] | None) -> str:
    """Why the backend does not build lr_scheduler (``_RunSettings.scheduler_unused``)."""
    if optimizer_type.lower().endswith("schedulefree"):  # the backend's own test
        return "schedule-free"
    if {**form, **user}.get("lr_scheduler_type"):
        return "class"
    if optimizer_type.lower() == "adafactor" and adafactor_relative_step(merged_args):
        return "adafactor"
    return ""


def _resolve_settings(
    preset: dict, form: dict, user: dict, *, optimizer: str, optimizer_args: str, optimizer_recommended: bool
) -> _RunSettings:
    """Work out a run's optimizer, scheduler and text-encoder settings, without changing anything.

    Only an optimizer_args entry without "=" raises.
    """
    values = {**preset, **form, **user}
    runs = optimizer_that_runs(optimizer, user, preset)
    friendly = _friendly_optimizer(runs) if runs else None
    if runs:
        optimizer_type = OPTIMIZER_NAMES.get(friendly, runs) if friendly else runs
    else:
        optimizer_type = str(values.get("optimizer_type", ""))
    recommended = OPTIMIZER_DEFAULTS.get(friendly, {}) if friendly and optimizer_recommended else {}
    follows_lr, replaced, rates = _learning_rates(values, user, recommended)
    merged_args = _merged_optimizer_args(preset, user, recommended, optimizer_type, optimizer_args)
    scheduler_unused = _scheduler_unused(optimizer_type, form, user, merged_args)
    # A rate of 0 does not freeze the text encoder: only network_train_unet_only does.
    # Read after the recommended settings, because Prodigy's set the rates to 1.0.
    zero_te_rate = rates.get("text_encoder_lr") == 0 and not values.get("network_train_unet_only")
    return _RunSettings(
        optimizer=runs,
        friendly=friendly,
        optimizer_type=optimizer_type,
        recommended=recommended,
        replaced=replaced,
        follows_lr=follows_lr,
        optimizer_args=merged_args,
        scheduler=str(values.get("lr_scheduler") or "constant"),
        scheduler_unused=scheduler_unused,
        zero_te_rate=zero_te_rate,
    )

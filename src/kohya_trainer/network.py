"""Network types: their backend modules, default sizes and adapter switches,
and the build_plan steps that set network_args, dim and alpha."""

from __future__ import annotations

from . import models
from .registry import ModelRoute

# LoHa and LoKr are built into the backend (networks/loha.py, networks/lokr.py).
# LoCon is networks.lora with conv layers, which do nothing on DiT routes.
NETWORK_MODULES = {
    "lora": "networks.lora",
    "locon": "networks.lora",
    "loha": "networks.loha",
    "lokr": "networks.lokr",
}

# The adapter switches each network module reads (the kwargs.get calls in its
# create_network), each on only for "true" in any case. A switch it does not
# read trains nothing without a word, so build_plan refuses it. A test checks the pin.
ADAPTER_ARGS_READ = {
    "networks.lora_anima": ("train_llm_adapter",),
    "networks.lora_generic": ("train_text_embedder", "train_llm_adapter"),
    "networks.loha": ("train_llm_adapter",),
    "networks.lokr": ("train_llm_adapter",),
}
ADAPTER_SWITCHES = frozenset(name for names in ADAPTER_ARGS_READ.values() for name in names)
# The adapter's learning rate, from the upstream Anima and MOSS-Image docs. Text, as the docs write it.
ADAPTER_LR = "5e-5"

# Defaults for each network type. They fill only what the caller did not set.
# - locon: conv_dim no larger than dim, with a low conv alpha, for a character.
# - loha: capacity grows with the square of dim, so a LoRA-sized dim memorizes
#   the images. Alpha stays at half of dim.
# - lokr: full-matrix mode, where `factor` sets the size. dim 10000 is a
#   threshold, not a size: it puts every layer in full-matrix mode. The
#   backend forces alpha = dim, so the scale is 1.0 (networks/lokr.py),
#   which ComfyUI expects. Compact mode below half the layer width learns
#   too little. Factor 8 works best, 4 overfits, and the file shrinks with
#   the square of the factor (LyCORIS guidelines, ai-toolkit).
NETWORK_DEFAULTS: dict[str, dict] = {
    "locon": {"network_args": ("conv_dim=8", "conv_alpha=4")},
    "loha": {"network_dim": 16, "network_alpha": 8},
    "lokr": {"network_dim": 10000, "network_alpha": 1, "network_args": ("factor=8",)},
}


def network_module_that_runs(network_type: str, config_overrides: dict, preset: dict) -> str:
    """The network module the run trains with: one in config_overrides wins over the type's."""
    if config_overrides.get("network_module"):
        return str(config_overrides["network_module"])
    if network_type != "lora":
        return NETWORK_MODULES.get(network_type, "")
    return str(preset.get("network_module") or "networks.lora")


def check_adapter_args(route: ModelRoute, network_module: str, network_args) -> list[str]:
    """Return the adapter switches on in network_args, and refuse one the run ignores.

    The backend turns a switch on only when the text after "=" is "true", in any
    case, with no spaces. The route and module checks come first, so the first
    message names the one change the run needs.
    """
    on, misread = [], []
    for name, _, value in (str(arg).partition("=") for arg in network_args or []):
        if name.strip() not in ADAPTER_SWITCHES:
            continue
        if name == name.strip() and value.lower() == "true":
            on.append(name)
        elif value.strip().lower() in ("true", "1", "yes", "on"):
            misread.append(f"{name}={value}")
    meant = on + [entry.partition("=")[0].strip() for entry in misread]
    if meant and not route.adapter_args:
        raise ValueError(
            f"{meant[0]}=True in network_args trains an adapter that {route.display_name} does not have, so it "
            "does nothing. Remove it from network_args."
        )
    reads = ADAPTER_ARGS_READ.get(network_module)
    if misread and reads is None:
        _refuse_misread(misread[0])
    if not meant or reads is None:
        return on
    for name in meant:
        if name not in route.adapter_args or name not in reads:
            right = [option for option in route.adapter_args if option in reads]
            way_out = (
                f'Use "{right[0]}=True" in network_args instead.'
                if right
                else f"{network_module} cannot train it, so pick another network type."
            )
            raise ValueError(
                f"{name}=True in network_args does nothing with {network_module} on {route.display_name}, so "
                f"{route.adapter_name} does not train. {way_out}"
            )
    if misread:
        _refuse_misread(misread[0])
    return on


def _refuse_misread(entry: str) -> None:
    name = entry.partition("=")[0]
    raise ValueError(
        f'"{entry}" in network_args reads as off, because the trainer turns it on only for the word true '
        f'right after the = sign. Write "{name.strip()}=True" without spaces, or remove it.'
    )


def _check_network_type(route: ModelRoute, network_type: str) -> None:
    """Refuse a network type that is unknown, or that the route cannot train."""
    if network_type not in NETWORK_MODULES:
        raise ValueError(f"unknown network_type {network_type!r}. The known types are {sorted(NETWORK_MODULES)}.")
    if network_type not in route.network_types:
        raise ValueError(
            f"route {route.key!r} supports network types {route.network_types}. "
            f"The pinned backend cannot train {network_type!r} on this model."
        )


def _has_arg(network_args: list[str], name: str) -> bool:
    return any(arg.partition("=")[0] == name for arg in network_args)


def _network_args(run_overrides: dict, network_type: str, conv_dim: int | None, conv_alpha: float | None) -> list[str]:
    """The run's network_args: the user's, the form's conv fields, then the type's defaults."""
    network_args = [str(arg) for arg in run_overrides.get("network_args") or []]
    # Before the type defaults, so the form's conv fields replace them.
    if network_type == "locon":
        if conv_dim is not None and not _has_arg(network_args, "conv_dim"):
            network_args.append(f"conv_dim={conv_dim}")
        if conv_alpha is not None and not _has_arg(network_args, "conv_alpha"):
            network_args.append(f"conv_alpha={conv_alpha:g}")
    for default_arg in NETWORK_DEFAULTS.get(network_type, {}).get("network_args", ()):
        if not _has_arg(network_args, default_arg.partition("=")[0]):
            network_args.append(default_arg)
    if network_args:
        run_overrides["network_args"] = network_args
    return network_args


def _network_size(run_overrides: dict, network_type: str, network_dim: int | None, network_alpha: float | None) -> None:
    """Set dim and alpha: the type's own sizes, or the form's fields for lora and locon."""
    scalar_defaults = {
        key: value for key, value in NETWORK_DEFAULTS.get(network_type, {}).items() if key != "network_args"
    }
    if not scalar_defaults:
        if network_dim is not None:
            run_overrides.setdefault("network_dim", network_dim)
        if network_alpha is not None:
            run_overrides.setdefault("network_alpha", network_alpha)
        return
    for key, value in scalar_defaults.items():
        run_overrides.setdefault(key, value)
    capacity = (
        # The backend ignores LoKr's alpha, so the note must not say "alpha 1".
        "a full matrix, sized by factor"
        if network_type == "lokr"
        else f"dim {run_overrides.get('network_dim')} / alpha {run_overrides.get('network_alpha')}"
    )
    change = '"network_args": ["factor=N"]' if network_type == "lokr" else "network_dim and network_alpha"
    print(
        f"{network_type} uses its own size settings: {capacity}. The form's network_dim and network_alpha "
        f"fields apply only to lora and locon. To change the size, add {change} to config_overrides."
    )
    if network_type == "lokr":
        print(
            "The backend warns about full matrix mode once for each layer. That is expected, so only "
            "the first warning is shown."
        )


def _adapter_rate(
    run_overrides: dict, route: ModelRoute, preset: dict, network_type: str, network_args: list[str]
) -> None:
    """Refuse an adapter switch the run ignores, and give a training adapter its lower learning rate."""
    module = network_module_that_runs(network_type, run_overrides, preset)
    adapter_on = check_adapter_args(route, module, network_args)
    if adapter_on and route.adapter_pattern and not _has_arg(network_args, "network_reg_lrs"):
        network_args.append(f"network_reg_lrs={route.adapter_pattern}={ADAPTER_LR}")
        run_overrides["network_args"] = network_args
        print(
            f"{adapter_on[0]} is on, so {route.adapter_name}'s learning rate is set to {ADAPTER_LR}, "
            "as the upstream doc advises"
        )


def _network_weights(
    run_overrides: dict, route: ModelRoute, network_type: str, network_args: list[str], network_weights: str, resume
) -> None:
    """Start from a finished LoRA file, without its optimizer state.

    dim_from_weights reads dim and alpha from the file (create_network_from_weights).
    LoKr's factor is not in the file, so a different factor fails to load.
    """
    weights_path = models.resolve_network_weights(str(network_weights), route.key)
    run_overrides.setdefault("network_weights", str(weights_path))
    run_overrides.setdefault("dim_from_weights", True)
    if network_type == "lokr":
        factor = next((a.partition("=")[2] for a in network_args if a.partition("=")[0] == "factor"), "-1")
        print(
            f"continuing from {weights_path.name}. Dim and alpha come from the file, but LoKr's factor "
            f"does not. This run uses factor={factor}, and a LoKr trained with a different factor will "
            'fail to load. To match it, add "network_args": ["factor=N"] to config_overrides.'
        )
    else:
        print(
            f"continuing from {weights_path.name}. Dim and alpha come from the file, so the form's fields are not used."
        )
    if resume:
        print("note: resume_from_state is also on. If a saved state is found, it is used instead of network_weights.")

"""The module the notebooks import.

The code lives in ``plan`` and the modules it uses. This module re-exports
the names that notebooks, scripts and tests reach through ``train.``.
"""

from __future__ import annotations

from . import dataset as dataset_mod
from . import environment, models, pins, tagging
from .dataset_checks import DATASET_FALLBACK_ARGS, dataset_override_keys
from .network import ADAPTER_ARGS_READ, NETWORK_DEFAULTS, NETWORK_MODULES, network_module_that_runs
from .optimizers import (
    OPTIMIZER_NAMES,
    OPTIMIZERS,
    SCHEDULERS,
    TRANSFORMERS_SCHEDULERS,
    backend_optimizer_names,
    check_scheduler,
    parse_optimizer_args,
    pinned_packages,
    preset_optimizer_args,
)
from .plan import (
    LaunchPlan,
    build_plan,
    check_before_downloads,
    check_project_name,
    execute,
    quick_train,
    record_resume_guard,
)
from .previews import default_sample_prompt, sample_size
from .resume import find_latest_state
from .settings import BLOCK_SWAP_CONFLICTS, check_block_swap, form_settings

__all__ = [
    "ADAPTER_ARGS_READ",
    "BLOCK_SWAP_CONFLICTS",
    "DATASET_FALLBACK_ARGS",
    "NETWORK_DEFAULTS",
    "NETWORK_MODULES",
    "OPTIMIZERS",
    "OPTIMIZER_NAMES",
    "SCHEDULERS",
    "TRANSFORMERS_SCHEDULERS",
    "LaunchPlan",
    "backend_optimizer_names",
    "build_plan",
    "check_before_downloads",
    "check_block_swap",
    "check_project_name",
    "check_scheduler",
    "dataset_mod",
    "dataset_override_keys",
    "default_sample_prompt",
    "environment",
    "execute",
    "find_latest_state",
    "form_settings",
    "models",
    "network_module_that_runs",
    "parse_optimizer_args",
    "pinned_packages",
    "pins",
    "preset_optimizer_args",
    "quick_train",
    "record_resume_guard",
    "sample_size",
    "tagging",
]

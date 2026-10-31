"""Resume: the saved state, the epoch to continue with, and the guard against a state that cannot load."""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import pins
from .registry import BaseModel


def _state_dir_complete(state_dir: Path) -> bool:
    # A session that ends mid-save leaves a partial folder (accelerator.save_state writes files one by one).
    return (state_dir / "train_state.json").is_file() and any(state_dir.glob("optimizer*"))


# The backend names step states {name}-step{:08d}-state. A final state is
# {name}-state, so a project called "x-step10" must not match.
_STEP_STATE = re.compile(r"-step\d{8,}-state$")


def _saved_epoch(state_dir: Path) -> int | None:
    """The epoch a saved state was in, from the backend's train_state.json."""
    try:
        data = json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    epoch = data.get("current_epoch") if isinstance(data, dict) else None
    return epoch if isinstance(epoch, int) and epoch > 0 else None


def find_latest_state(output_dir: Path, output_name: str) -> Path | None:
    """The newest complete sd-scripts state folder for this run, or None.

    If no folder looks complete, the newest is returned, because a later
    backend can rename its files.
    """
    states = sorted(
        (p for p in Path(output_dir).glob(f"{output_name}*-state") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not states:
        return None
    complete = [s for s in states if _state_dir_complete(s)]
    return complete[0] if complete else states[0]


def _resume_settings(
    run_overrides: dict, resume: bool | str | Path, output_dir: Path, project_name: str
) -> tuple[int | None, str]:
    """Find the state to resume from. Returns the epoch to continue with (or None), and a note on why."""
    continue_epoch, continue_note = None, ""
    # config_overrides can move the output folder and rename the files.
    run_output_dir = Path(run_overrides.get("output_dir") or output_dir)
    run_name = str(run_overrides.get("output_name") or project_name)
    state_dir = find_latest_state(run_output_dir, run_name) if resume is True else Path(resume)
    if state_dir and state_dir.is_dir():
        run_overrides.setdefault("resume", str(state_dir))
        print(f"resuming from {state_dir}")
        # The backend resets its step count on resume, so a second resume
        # starts too early and overwrites files. The saved epoch stays right.
        epoch = _saved_epoch(state_dir)
        if epoch and "initial_epoch" not in run_overrides and "initial_step" not in run_overrides:
            # An epoch or final state is saved after its epoch ends. A step state can be from partway through it.
            if _STEP_STATE.search(state_dir.name):
                continue_epoch = epoch
                continue_note = f": a step state can be from partway through epoch {epoch}, so it runs again"
            else:
                continue_epoch = epoch + 1
            run_overrides["initial_epoch"] = continue_epoch
    elif resume is True:
        print("resume is on, but no saved state was found, so training starts fresh")
    else:
        raise FileNotFoundError(f"resume state dir not found: {resume}")
    return continue_epoch, continue_note


def _check_epochs_left(continue_epoch: int | None, train_config: dict) -> None:
    """Refuse a resume with no epoch left to train. The backend refuses it only after loading the models."""
    epochs_planned = train_config.get("max_train_epochs")
    if continue_epoch and epochs_planned and continue_epoch > epochs_planned:
        raise ValueError(
            f"the saved state continues with epoch {continue_epoch}, but this run is set to "
            f"{epochs_planned} epochs, so there is nothing left to train. Raise max_train_epochs "
            "to train further, or start a new project."
        )


def _resume_guard(train_config: dict, network_type: str, base: BaseModel, guard_file: Path) -> dict:
    """Return what this run trains with, and refuse a resume whose saved state cannot load.

    A changed network shape fails deep in the backend with a shape error. lora
    and locon share networks.lora, and conv_dim, conv_alpha and LoKr's factor
    live in network_args, so the type and every name=value entry are recorded too.
    """
    guard = {key: train_config.get(key) for key in ("network_dim", "network_alpha", "network_module")}
    guard["network_type"] = network_type
    guard["network_args"] = sorted(str(arg) for arg in train_config.get("network_args") or [] if "=" in str(arg))
    guard["base_key"] = base.key
    if train_config.get("resume") and guard_file.is_file():
        previous = json.loads(guard_file.read_text(encoding="utf-8"))
        changed = {k: (previous[k], guard[k]) for k in guard if k in previous and previous[k] != guard[k]}
        if changed:
            details = "".join(f"- {k} was {old!r}, now {new!r}\n" for k, (old, new) in sorted(changed.items()))
            raise ValueError(
                f"resume with changed settings:\n{details}The saved state cannot load into a different "
                "network or base. Restore the previous settings, or start a fresh project."
            )
        # A state from another backend version usually loads, so only a note.
        saved_with = previous.get("sd_scripts_commit")
        if saved_with is None:
            print(
                "note: this state was saved by an older version of this trainer, so its sd-scripts version "
                "is unknown. If loading the state fails, start a fresh project."
            )
        elif saved_with != pins.SD_SCRIPTS_COMMIT:
            print(
                f"note: this state was saved with sd-scripts {saved_with[:7]}, and this trainer now uses "
                f"{pins.SD_SCRIPTS_COMMIT[:7]}. Resuming usually works. If loading the state fails, "
                "start a fresh project."
            )
    guard["sd_scripts_commit"] = pins.SD_SCRIPTS_COMMIT
    return guard

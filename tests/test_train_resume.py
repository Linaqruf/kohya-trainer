"""Resume: finding saved states, the resume guard and the epoch a run continues with."""

from __future__ import annotations

import json
import sys

import pytest

from kohya_trainer import pins, train

from conftest import _fake_plan, _plan, needs_submodule


@needs_submodule
def test_preset_saves_resumable_state(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path)
    assert plan.train_config["save_state"] is True
    assert plan.train_config["save_last_n_epochs_state"] == 1


def test_find_latest_state_picks_newest(tmp_path):
    import os

    assert train.find_latest_state(tmp_path, "proj") is None
    old = tmp_path / "proj-000001-state"
    new = tmp_path / "proj-000003-state"
    old.mkdir()
    new.mkdir()
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    (tmp_path / "other-000009-state").mkdir()  # different project: ignored
    assert train.find_latest_state(tmp_path, "proj") == new


def _complete_state_dir(path):
    path.mkdir(parents=True)
    (path / "train_state.json").write_text("{}", encoding="utf-8")
    (path / "optimizer.bin").write_bytes(b"0")


def test_find_latest_state_skips_partial_dirs(tmp_path):
    """A crash during accelerator.save_state leaves a partial newest folder."""
    import os

    good = tmp_path / "proj-000002-state"
    _complete_state_dir(good)
    partial = tmp_path / "proj-000003-state"
    partial.mkdir()  # no marker files: interrupted save
    os.utime(good, (1_000_000, 1_000_000))
    os.utime(partial, (2_000_000, 2_000_000))
    assert train.find_latest_state(tmp_path, "proj") == good
    # If no folder passes the check, the newest is used.
    (good / "train_state.json").unlink()
    (good / "optimizer.bin").unlink()
    os.utime(good, (1_000_000, 1_000_000))  # unlinking bumped the dir mtime
    assert train.find_latest_state(tmp_path, "proj") == partial


@needs_submodule
def test_resume_guard_rejects_changed_network(image_dataset, tmp_path, capsys):
    train.record_resume_guard(_plan(image_dataset, tmp_path))  # the first run trains
    state = tmp_path / "ws" / "output" / "test-lora-000001-state"
    _complete_state_dir(state)
    (state / "train_state.json").write_text(json.dumps({"current_epoch": 1, "current_step": 200}), encoding="utf-8")
    plan = _plan(image_dataset, tmp_path, resume=True)  # unchanged settings resume fine
    assert plan.train_config["resume"] == str(state)
    capsys.readouterr()
    with pytest.raises(ValueError, match="network_dim"):
        _plan(image_dataset, tmp_path, resume=True, config_overrides={"network_dim": 64})
    assert "continuing with epoch" not in capsys.readouterr().out  # never printed before a refusal
    # A fresh run (no resume) can change anything. Its record is written when it trains.
    plan = _plan(image_dataset, tmp_path, config_overrides={"network_dim": 64})
    assert plan.train_config["network_dim"] == 64
    train.record_resume_guard(plan)
    _plan(image_dataset, tmp_path, resume=True, config_overrides={"network_dim": 64})


@needs_submodule
def test_a_plan_built_only_to_look_keeps_the_record(image_dataset, tmp_path):
    """A plan at dim 16 built only to look must not hide that the state on disk is dim 32."""
    train.record_resume_guard(_plan(image_dataset, tmp_path, network_dim=32))
    _complete_state_dir(tmp_path / "ws" / "output" / "test-lora-000001-state")
    _plan(image_dataset, tmp_path, network_dim=16)
    with pytest.raises(ValueError, match="network_dim was 32, now 16"):
        _plan(image_dataset, tmp_path, network_dim=16, resume=True)


def test_execute_records_the_guard_when_training_starts(tmp_path):
    plan = _fake_plan([sys.executable, "-c", "print('ok')"], tmp_path)
    plan.resume_guard = {"network_dim": 8}
    plan.resume_guard_file = tmp_path / "config" / "resume_guard.json"
    train.execute(plan, sd_scripts_dir=tmp_path)
    assert json.loads(plan.resume_guard_file.read_text(encoding="utf-8")) == {"network_dim": 8}


@needs_submodule
@pytest.mark.parametrize(
    "first, then, named",
    [
        ({}, {"network_type": "locon"}, "network_type"),
        ({"network_type": "locon"}, {"network_type": "locon", "conv_dim": 16}, "conv_dim=16"),
        (
            {"network_type": "lokr"},
            {"network_type": "lokr", "config_overrides": {"network_args": ["factor=4"]}},
            "factor=4",
        ),
    ],
)
def test_resume_guard_covers_the_network_type_and_its_args(image_dataset, tmp_path, first, then, named):
    train.record_resume_guard(_plan(image_dataset, tmp_path, **first))
    _complete_state_dir(tmp_path / "ws" / "output" / "test-lora-000001-state")
    _plan(image_dataset, tmp_path, resume=True, **first)  # unchanged: resumes
    with pytest.raises(ValueError, match=named):
        _plan(image_dataset, tmp_path, resume=True, **then)


@needs_submodule
def test_resume_guard_from_before_the_type_was_recorded_still_resumes(image_dataset, tmp_path):
    train.record_resume_guard(_plan(image_dataset, tmp_path))
    _complete_state_dir(tmp_path / "ws" / "output" / "test-lora-000001-state")
    _rewrite_guard(tmp_path, lambda guard: [guard.pop("network_type"), guard.pop("network_args")])
    plan = _plan(image_dataset, tmp_path, resume=True, network_type="locon")
    assert plan.train_config["resume"].endswith("test-lora-000001-state")


@needs_submodule
def test_resume_finds_states_under_the_output_name_and_folder_the_run_uses(image_dataset, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    state = elsewhere / "mychar_v2-000002-state"
    _complete_state_dir(state)
    overrides = {"output_dir": str(elsewhere), "output_name": "mychar_v2"}
    plan = _plan(image_dataset, tmp_path, resume=True, config_overrides=overrides)
    assert plan.train_config["resume"] == str(state)
    assert plan.output_dir == elsewhere
    # The project's own folder has no state under that name.
    plan = _plan(image_dataset, tmp_path, resume=True, config_overrides={"output_name": "mychar_v2"})
    assert "resume" not in plan.train_config


def _rewrite_guard(tmp_path, change):
    guard_file = tmp_path / "ws" / "config" / "resume_guard.json"
    guard = json.loads(guard_file.read_text(encoding="utf-8"))
    change(guard)
    guard_file.write_text(json.dumps(guard), encoding="utf-8")


@needs_submodule
def test_resume_notes_a_state_saved_with_another_backend(image_dataset, tmp_path, capsys):
    train.record_resume_guard(_plan(image_dataset, tmp_path))
    guard_file = tmp_path / "ws" / "config" / "resume_guard.json"
    assert json.loads(guard_file.read_text(encoding="utf-8"))["sd_scripts_commit"] == pins.SD_SCRIPTS_COMMIT
    state = tmp_path / "ws" / "output" / "test-lora-000001-state"
    _complete_state_dir(state)
    capsys.readouterr()

    # Same backend: no note.
    _plan(image_dataset, tmp_path, resume=True)
    assert "sd-scripts" not in capsys.readouterr().out

    # Another backend: a note, and the plan is still built.
    _rewrite_guard(tmp_path, lambda guard: guard.update(sd_scripts_commit="0123456789abcdef"))
    plan = _plan(image_dataset, tmp_path, resume=True)
    out = capsys.readouterr().out
    assert plan.train_config["resume"] == str(state)
    assert f"saved with sd-scripts 0123456, and this trainer now uses {pins.SD_SCRIPTS_COMMIT[:7]}" in out
    assert "start a fresh project" in out

    # A guard from before the commit was recorded: the softer note.
    _rewrite_guard(tmp_path, lambda guard: guard.pop("sd_scripts_commit"))
    plan = _plan(image_dataset, tmp_path, resume=True)
    out = capsys.readouterr().out
    assert plan.train_config["resume"] == str(state)
    assert "older version of this trainer" in out
    train.record_resume_guard(plan)  # training starts, so the record is brought up to date
    assert json.loads(guard_file.read_text(encoding="utf-8"))["sd_scripts_commit"] == pins.SD_SCRIPTS_COMMIT


@needs_submodule
def test_resume_auto_finds_state_or_starts_fresh(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, resume=True)  # no state yet
    assert "resume" not in plan.train_config
    state = plan.output_dir / "test-lora-000001-state"
    state.mkdir()
    plan = _plan(image_dataset, tmp_path, resume=True)
    assert plan.train_config["resume"] == str(state)


@needs_submodule
def test_resume_continues_from_the_saved_epoch(image_dataset, tmp_path, capsys):
    """After one resume the backend's saved step count restarts at 0. A resume by the step count
    then starts at an earlier epoch and overwrites its file."""
    plan = _plan(image_dataset, tmp_path)
    state = plan.output_dir / "test-lora-000003-state"
    state.mkdir()
    (state / "train_state.json").write_text(json.dumps({"current_epoch": 3, "current_step": 200}), encoding="utf-8")
    plan = _plan(image_dataset, tmp_path, resume=True)
    assert plan.train_config["resume"] == str(state)
    assert plan.train_config["initial_epoch"] == 4
    assert "continuing with epoch 4" in capsys.readouterr().out
    # A choice the user made wins.
    plan = _plan(image_dataset, tmp_path, resume=True, config_overrides={"initial_epoch": 2})
    assert plan.train_config["initial_epoch"] == 2
    # No readable epoch: leave it to the backend.
    (state / "train_state.json").write_text("not json", encoding="utf-8")
    plan = _plan(image_dataset, tmp_path, resume=True)
    assert "initial_epoch" not in plan.train_config


@needs_submodule
def test_resume_from_a_step_state_runs_its_epoch_again(image_dataset, tmp_path, capsys):
    """A step state names the epoch it was saved in, which can be unfinished."""
    plan = _plan(image_dataset, tmp_path)
    state = plan.output_dir / "test-lora-step00000250-state"
    state.mkdir()
    (state / "train_state.json").write_text(json.dumps({"current_epoch": 2, "current_step": 250}), encoding="utf-8")
    capsys.readouterr()
    plan = _plan(image_dataset, tmp_path, resume=state)
    assert plan.train_config["initial_epoch"] == 2
    assert "continuing with epoch 2: a step state can be from partway through epoch 2" in capsys.readouterr().out
    # The last planned epoch is not refused.
    plan = _plan(image_dataset, tmp_path, resume=state, config_overrides={"max_train_epochs": 2})
    assert plan.train_config["initial_epoch"] == 2


@needs_submodule
def test_resume_reads_a_final_state_of_a_project_named_like_a_step(image_dataset, tmp_path):
    # The final state of project "x-step10" is "x-step10-state", not a step state.
    plan = _plan(image_dataset, tmp_path, project_name="x-step10")
    state = plan.output_dir / "x-step10-state"
    state.mkdir()
    (state / "train_state.json").write_text(json.dumps({"current_epoch": 3, "current_step": 600}), encoding="utf-8")
    plan = _plan(image_dataset, tmp_path, project_name="x-step10", resume=state)
    assert plan.train_config["initial_epoch"] == 4


@needs_submodule
def test_resume_of_a_finished_run_is_refused_early(image_dataset, tmp_path, capsys):
    plan = _plan(image_dataset, tmp_path, config_overrides={"max_train_epochs": 3})
    state = plan.output_dir / "test-lora-state"
    state.mkdir()
    (state / "train_state.json").write_text(json.dumps({"current_epoch": 3, "current_step": 600}), encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(ValueError, match="continues with epoch 4, but this run is set to 3 epochs"):
        _plan(image_dataset, tmp_path, resume=state, config_overrides={"max_train_epochs": 3})
    assert "continuing with epoch" not in capsys.readouterr().out
    # Raising the epoch count trains further.
    plan = _plan(image_dataset, tmp_path, resume=state, config_overrides={"max_train_epochs": 5})
    assert plan.train_config["initial_epoch"] == 4
    # Training by steps instead of epochs is left to the backend.
    plan = _plan(
        image_dataset, tmp_path, resume=state, config_overrides={"max_train_epochs": None, "max_train_steps": 900}
    )
    assert plan.train_config["initial_epoch"] == 4
    # An epoch the user chose is theirs, even past the planned count.
    plan = _plan(image_dataset, tmp_path, resume=state, config_overrides={"max_train_epochs": 3, "initial_epoch": 5})
    assert plan.train_config["initial_epoch"] == 5


@needs_submodule
def test_resume_explicit_path_must_exist(image_dataset, tmp_path):
    with pytest.raises(FileNotFoundError, match="state dir not found"):
        _plan(image_dataset, tmp_path, resume=tmp_path / "nope-state")

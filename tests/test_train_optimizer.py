"""Optimizers, schedulers and learning rates: the recommended settings, the checks and the layering."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from kohya_trainer import config, optimizers, registry, train

from conftest import ROUTE, SD_SCRIPTS, _anima_plan, _plan, _sd15_plan, needs_submodule


@needs_submodule
def test_optimizer_dropdown_recommended_settings(image_dataset, tmp_path):
    """Adaptive optimizers set their own rate, so the recommended settings replace the form's rate fields."""
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", form_values={"learning_rate": 3e-4})
    assert plan.train_config["optimizer_type"] == "Prodigy"
    assert plan.train_config["learning_rate"] == 1.0  # replaced, not only filled in
    assert "decouple=True" in plan.train_config["optimizer_args"]
    # The scheduler is not part of the recommended set: the preset's (or the form's) runs.
    assert plan.train_config["lr_scheduler"] == "cosine"
    # With the recommended settings off, the form's values stay.
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="Prodigy",
        optimizer_recommended=False,
        form_values={"learning_rate": 3e-4},
    )
    assert plan.train_config["learning_rate"] == 3e-4
    assert "optimizer_args" not in plan.train_config
    # Typed args win name by name, and the recommended set fills the rest.
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", optimizer_args="d_coef=1")
    args = plan.train_config["optimizer_args"]
    assert "d_coef=1" in args and "d_coef=2" not in args
    assert "decouple=True" in args  # the other recommended entries stay
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", optimizer_recommended=False, optimizer_args="d_coef=1")
    assert plan.train_config["optimizer_args"] == ["d_coef=1"]
    # The default optimizer changes nothing but optimizer_type.
    plan = _plan(image_dataset, tmp_path, optimizer="AdamW8bit")
    assert plan.train_config["optimizer_type"] == "AdamW8bit"
    assert plan.train_config["learning_rate"] == 3e-4  # preset value untouched
    with pytest.raises(ValueError, match="Did you mean AdamW8bit"):
        _plan(image_dataset, tmp_path, optimizer="AdamW9bit")


@needs_submodule
def test_settings_layer_preset_form_recommended_then_overrides(image_dataset, tmp_path):
    """preset < form values < recommended optimizer settings < config_overrides."""
    sd15 = dict(form_values={"learning_rate": 2e-4, "text_encoder_lr": 5e-5, "seed": 7})
    plan = _sd15_plan(image_dataset, tmp_path, **sd15)
    assert plan.train_config["learning_rate"] == 2e-4
    assert plan.train_config["unet_lr"] == 2e-4  # the preset's unet_lr follows the form's learning rate
    assert plan.train_config["text_encoder_lr"] == 5e-5
    assert plan.train_config["seed"] == 7
    # The recommended settings replace the form's values...
    plan = _sd15_plan(image_dataset, tmp_path, optimizer="Prodigy", **sd15)
    assert plan.train_config["learning_rate"] == plan.train_config["unet_lr"] == 1.0
    assert plan.train_config["seed"] == 7  # only the keys they name
    # ...and lose to what the user put in config_overrides.
    plan = _sd15_plan(
        image_dataset, tmp_path, optimizer="Prodigy", config_overrides={"learning_rate": 0.5, "unet_lr": 0.5}, **sd15
    )
    assert plan.train_config["learning_rate"] == plan.train_config["unet_lr"] == 0.5
    # The text encoder follows the user's learning_rate, not Prodigy's 1.0.
    assert plan.train_config["text_encoder_lr"] == 0.5
    # config_overrides wins over the form, and a None removes the key.
    plan = _sd15_plan(image_dataset, tmp_path, config_overrides={"seed": 9, "clip_skip": None}, **sd15)
    assert plan.train_config["seed"] == 9
    assert "clip_skip" not in plan.train_config


@needs_submodule
@pytest.mark.parametrize("optimizer", ["", "AdamW8bit"])
def test_a_learning_rate_in_config_overrides_reaches_the_unet_on_sd15(image_dataset, tmp_path, capsys, optimizer):
    """The backend lets unet_lr win over learning_rate, and SD1.5's preset and
    form set unet_lr, so the user's learning_rate must set it too."""
    form = {"learning_rate": 5e-4, "text_encoder_lr": 1e-4}
    plan = _sd15_plan(
        image_dataset, tmp_path, optimizer=optimizer, form_values=form, config_overrides={"learning_rate": 2e-4}
    )
    assert plan.train_config["learning_rate"] == plan.train_config["unet_lr"] == 2e-4
    assert plan.train_config["text_encoder_lr"] == 1e-4  # the form's own field stays
    assert "also sets unet_lr" in capsys.readouterr().out
    # With no form values the preset's unet_lr follows it too.
    plan = _sd15_plan(image_dataset, tmp_path, optimizer=optimizer, config_overrides={"learning_rate": 2e-4})
    assert plan.train_config["unet_lr"] == 2e-4


@needs_submodule
def test_a_learning_rate_in_config_overrides_wins_over_prodigys_rates(image_dataset, tmp_path, capsys):
    form = {"learning_rate": 5e-4, "text_encoder_lr": 1e-4}
    plan = _sd15_plan(
        image_dataset, tmp_path, optimizer="Prodigy", form_values=form, config_overrides={"learning_rate": 0.5}
    )
    config = plan.train_config
    assert config["learning_rate"] == config["unet_lr"] == config["text_encoder_lr"] == 0.5
    out = capsys.readouterr().out
    assert "replaces the form's learning rate" not in out
    assert "also sets unet_lr and text_encoder_lr" in out
    assert "decouple=True" in out  # the recommended optimizer_args still apply
    # SDXL trains the U-Net only. The U-Net runs at the user's rate.
    sdxl = registry.get("sdxl")
    plan = train.build_plan(
        sdxl,
        sdxl.base(),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "wsxl",
        project_name="xl",
        sd_scripts_dir=SD_SCRIPTS,
        optimizer="Prodigy",
        form_values={"learning_rate": 3e-4},
        config_overrides={"learning_rate": 0.5},
    )
    assert plan.train_config["learning_rate"] == plan.train_config["unet_lr"] == 0.5
    # A unet_lr of their own still wins.
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", config_overrides={"learning_rate": 0.5, "unet_lr": 0.8})
    assert plan.train_config["unet_lr"] == 0.8


@needs_submodule
def test_form_zero_turns_a_number_off(image_dataset, tmp_path):
    form = {
        "max_train_epochs": 10,
        "max_train_steps": 0,
        "gradient_accumulation_steps": 0,
        "save_last_n_epochs": 0,
        "min_snr_gamma": 0,
        "noise_offset": 0,
    }
    plan = _plan(image_dataset, tmp_path, form_values=form)
    config = plan.train_config
    assert config["max_train_epochs"] == 10 and "max_train_steps" not in config
    for key in ("gradient_accumulation_steps", "save_last_n_epochs", "min_snr_gamma", "noise_offset"):
        assert key not in config, key
    # Steps above 0 train by steps.
    plan = _plan(image_dataset, tmp_path, form_values={**form, "max_train_steps": 500})
    assert plan.train_config["max_train_steps"] == 500 and "max_train_epochs" not in plan.train_config
    assert plan.estimated_total_steps == 500
    # Steps in config_overrides win over the form's epochs too. The backend lets epochs win, so the plan drops them.
    plan = _plan(image_dataset, tmp_path, form_values=form, config_overrides={"max_train_steps": 300})
    assert plan.train_config["max_train_steps"] == 300 and "max_train_epochs" not in plan.train_config


@needs_submodule
def test_optimizer_note_names_the_form_field_and_says_what_it_replaces(image_dataset, tmp_path, capsys):
    _plan(image_dataset, tmp_path, optimizer="Prodigy")
    out = capsys.readouterr().out
    assert "a learning rate of 1, which replaces the form's learning rate" in out
    assert "config_overrides" in out
    assert "apply_optimizer_defaults" in out and "optimizer_recommended" not in out
    # AdaFactor's recommended set is only optimizer_args: nothing else is replaced.
    _plan(image_dataset, tmp_path, optimizer="AdaFactor")
    out = capsys.readouterr().out
    assert "settings in optimizer_args" in out and "apply_optimizer_defaults" in out
    assert "learning rate" not in out


@needs_submodule
def test_constant_scheduler_drops_warmup(image_dataset, tmp_path):
    """The backend refuses warmup with plain constant after latent caching, and the advanced
    form always sends the preset's warmup ratio."""
    plan = _plan(image_dataset, tmp_path, config_overrides={"lr_scheduler": "constant", "lr_warmup_steps": 0.05})
    assert "lr_warmup_steps" not in plan.train_config
    # A scheduler that takes warmup keeps it.
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", config_overrides={"lr_warmup_steps": 0.05})
    assert plan.train_config["lr_warmup_steps"] == 0.05
    # Picking constant drops Anima's 100-step warmup, and a form-sent 0.0 too.
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"lr_scheduler": "constant"})
    assert "lr_warmup_steps" not in plan.train_config
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"lr_scheduler": "constant", "lr_warmup_steps": 0.0})
    assert "lr_warmup_steps" not in plan.train_config
    # The preset's own scheduler keeps its warmup.
    plan = _anima_plan(image_dataset, tmp_path)
    assert plan.train_config["lr_scheduler"] == "constant_with_warmup"
    assert plan.train_config["lr_warmup_steps"] == 100


@needs_submodule
def test_constant_scheduler_names_the_warmup_scheduler_when_it_drops_one(image_dataset, tmp_path, capsys):
    _plan(image_dataset, tmp_path, form_values={"lr_scheduler": "constant", "lr_warmup_steps": 100})
    assert "pick constant_with_warmup" in capsys.readouterr().out
    # A warmup of 0 (the Anima form's default) is dropped without a word.
    _anima_plan(image_dataset, tmp_path, form_values={"lr_scheduler": "constant", "lr_warmup_steps": 0.0})
    assert "constant_with_warmup" not in capsys.readouterr().out


@needs_submodule
def test_prodigy_runs_the_scheduler_on_the_form(image_dataset, tmp_path):
    form = {"lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 0.05}
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", form_values=form)
    assert plan.train_config["lr_scheduler"] == "constant_with_warmup"
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", form_values={**form, "lr_scheduler": "linear"})
    assert plan.train_config["lr_scheduler"] == "linear"
    assert plan.train_config["learning_rate"] == 1.0


@needs_submodule
def test_recommended_settings_follow_the_optimizer_that_runs(image_dataset, tmp_path):
    """An optimizer_type in config_overrides wins over the dropdown, and the
    recommended settings are those of the optimizer that runs."""
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="AdamW8bit",
        form_values={"learning_rate": 3e-4},
        config_overrides={"optimizer_type": "Prodigy"},
    )
    assert plan.train_config["optimizer_type"] == "Prodigy"
    assert plan.train_config["learning_rate"] == 1.0
    assert "d_coef=2" in plan.train_config["optimizer_args"]
    # The backend lowercases plain names, so the recommended set does too.
    plan = _plan(image_dataset, tmp_path, config_overrides={"optimizer_type": "prodigy"})
    assert plan.train_config["learning_rate"] == 1.0
    # A friendly name typed in config_overrides runs its backend class.
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", config_overrides={"optimizer_type": "CAME"})
    assert plan.train_config["optimizer_type"] == "pytorch_optimizer.CAME"
    assert plan.train_config["optimizer_args"] == ["weight_decay=0.04"]
    assert plan.train_config["learning_rate"] == 3e-4


@needs_submodule
def test_came_and_prodigy_schedule_free_run_their_backend_classes(image_dataset, tmp_path, capsys):
    plan = _plan(image_dataset, tmp_path, optimizer="CAME", form_values={"learning_rate": 1.5e-4})
    assert plan.train_config["optimizer_type"] == "pytorch_optimizer.CAME"
    assert plan.train_config["optimizer_args"] == ["weight_decay=0.04"]
    assert plan.train_config["learning_rate"] == 1.5e-4  # CAME keeps the form's rate
    assert "weight_decay=0.04" in capsys.readouterr().out
    form = {"learning_rate": 3e-4, "lr_scheduler": "constant", "lr_warmup_steps": 0.05}
    plan = _sd15_plan(image_dataset, tmp_path, optimizer="ProdigyScheduleFree", form_values=form)
    config = plan.train_config
    assert config["optimizer_type"] == "prodigyplus.ProdigyPlusScheduleFree"
    assert config["learning_rate"] == config["unet_lr"] == config["text_encoder_lr"] == 1.0
    out = capsys.readouterr().out
    assert "schedule-free" in out and "does not use lr_scheduler or lr_warmup_steps" in out
    # The backend makes a dummy scheduler for it and checks no warmup, so nothing is rewritten.
    assert config["lr_warmup_steps"] == 0.05
    # Its package says not to clip gradients, so the recommended settings turn clipping off...
    assert config["max_grad_norm"] == 0.0
    assert "max_grad_norm 0, which turns off gradient clipping" in out
    # ...unless the user sets max_grad_norm.
    plan = _sd15_plan(
        image_dataset,
        tmp_path,
        optimizer="ProdigyScheduleFree",
        form_values=form,
        config_overrides={"max_grad_norm": 0.5},
    )
    assert plan.train_config["max_grad_norm"] == 0.5
    assert "gradient clipping" not in capsys.readouterr().out


@needs_submodule
@pytest.mark.parametrize(
    "name",
    ["Lion8bit", "Lion", "PagedAdamW8bit", "AdamWScheduleFree", "RAdamScheduleFree", "SGD", "adamw", "AdaFactor"],
)
def test_typed_optimizers_the_backend_knows_pass(image_dataset, tmp_path, name):
    plan = _plan(image_dataset, tmp_path, optimizer=name)
    assert plan.train_config["optimizer_type"] == name


@needs_submodule
def test_optimizer_typos_are_refused_before_caching(image_dataset, tmp_path):
    with pytest.raises(ValueError, match="Did you mean Lion8bit"):
        _plan(image_dataset, tmp_path, optimizer="Lion8bt")
    with pytest.raises(ValueError, match="no package 'nosuchpackage' is installed"):
        _plan(image_dataset, tmp_path, config_overrides={"optimizer_type": "nosuchpackage.optim.Fancy"})
    # A dotted name whose class exists passes. The backend imports it.
    plan = _plan(image_dataset, tmp_path, optimizer="collections.OrderedDict")
    assert plan.train_config["optimizer_type"] == "collections.OrderedDict"


@needs_submodule
def test_backend_names_whose_package_is_not_installed_are_refused(image_dataset, tmp_path, monkeypatch):
    """The backend knows DAdaptation, but its requirements do not install dadaptation.
    The check reads requirements.txt, so it works without the training packages."""
    monkeypatch.setattr(optimizers.importlib.util, "find_spec", lambda name, *a: None)  # nothing installed
    for name in ("DAdaptation", "DAdaptAdam", "dadaptlion"):
        with pytest.raises(ValueError, match="needs the dadaptation package"):
            _plan(image_dataset, tmp_path, optimizer=name)
    # The packages the requirements do install pass.
    for name in ("Prodigy", "Lion", "PagedAdamW8bit", "AdamWScheduleFree", "AdamW8bit"):
        _plan(image_dataset, tmp_path, optimizer=name)
    # A typo is never pointed at a name that cannot run.
    with pytest.raises(ValueError, match="not one the backend knows") as error:
        _plan(image_dataset, tmp_path, optimizer="DAdaptAdm")
    assert "DAdaptAdam" not in str(error.value)


@needs_submodule
def test_a_package_optimizer_is_checked_down_to_its_class(image_dataset, tmp_path, monkeypatch):
    """When the package imports here, a misspelled class is refused with the
    close names. When it does not, the package must be in the pinned requirements."""
    package = tmp_path / "pkg" / "fakeoptimizers"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("class FancyAdam:\n    pass\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "pkg"))
    assert _plan(image_dataset, tmp_path, optimizer="fakeoptimizers.FancyAdam").train_config["optimizer_type"]
    with pytest.raises(ValueError, match="has no 'FancyAdan'.*Did you mean fakeoptimizers.FancyAdam"):
        _plan(image_dataset, tmp_path, optimizer="fakeoptimizers.FancyAdan")
    with pytest.raises(ValueError, match="cannot be imported"):
        _plan(image_dataset, tmp_path, optimizer="fakeoptimizers.nosub.Fancy")
    # Without the training packages (as in CI), a pinned package is enough...
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        optimizers.importlib.util,
        "find_spec",
        lambda name, *a: None if name in ("pytorch_optimizer", "prodigyplus") else real_find_spec(name, *a),
    )
    for name in ("pytorch_optimizer.Lamb", "prodigyplus.ProdigyPlusScheduleFree"):
        assert _plan(image_dataset, tmp_path, optimizer=name).train_config["optimizer_type"] == name
    # ...and a package it does not pin is refused.
    with pytest.raises(ValueError, match="no package 'pytorch_optimiser'"):
        _plan(image_dataset, tmp_path, optimizer="pytorch_optimiser.CAME")


@needs_submodule
def test_backend_optimizer_names_come_from_the_pinned_source():
    names = train.backend_optimizer_names(str(SD_SCRIPTS))
    for name in ("AdamW8bit", "AdamW", "Lion8bit", "PagedAdamW8bit", "Prodigy", "Adafactor", "RAdamScheduleFree"):
        assert name in names, name
    assert "CAME" not in names  # CAME is loaded as pytorch_optimizer.CAME


@needs_submodule
def test_listed_package_optimizers_are_in_the_pinned_requirements():
    """Each dotted optimizer we list comes from a package the backend pins, so setup installs it."""
    pinned = train.pinned_packages(str(SD_SCRIPTS))
    assert {"pytorch_optimizer", "prodigy_plus_schedule_free", "bitsandbytes", "prodigyopt"} <= pinned
    assert "dadaptation" not in pinned
    for friendly in train.OPTIMIZERS:
        name = train.OPTIMIZER_NAMES.get(friendly, friendly)
        if "." in name:
            module = name.partition(".")[0]
            assert optimizers._IMPORT_PACKAGES.get(module, module) in pinned, name
    assert "Lion8bit" not in train.OPTIMIZERS


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        ("weight_decay=0.05", ["weight_decay=0.05"]),
        ("betas=0.9, 0.999 weight_decay=0.05", ["betas=0.9,0.999", "weight_decay=0.05"]),
        ("betas=0.9 ,0.999", ["betas=0.9,0.999"]),
        ("weight_decay=0.01, decouple=True", ["weight_decay=0.01", "decouple=True"]),
    ],
)
def test_optimizer_args_field_keeps_values_with_spaces_together(text, expected):
    assert train.parse_optimizer_args(text) == expected


def test_optimizer_args_field_refuses_a_value_without_a_name():
    with pytest.raises(ValueError, match="name=value"):
        train.parse_optimizer_args("0.9")


@needs_submodule
def test_optimizer_args_merge_field_and_overrides_by_name(image_dataset, tmp_path):
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="Prodigy",
        optimizer_args="weight_decay=0.05 d_coef=1",
        config_overrides={"optimizer_args": ["d_coef=3", "growth_rate=1.02"]},
    )
    args = plan.train_config["optimizer_args"]
    assert "d_coef=3" in args and "d_coef=1" not in args and "d_coef=2" not in args
    assert "weight_decay=0.05" in args and "weight_decay=0.01" not in args
    assert "decouple=True" in args and "growth_rate=1.02" in args
    # The field and the config_overrides list merge.
    plan = _plan(
        image_dataset, tmp_path, optimizer_args="weight_decay=0.05", config_overrides={"optimizer_args": ["eps=1e-7"]}
    )
    assert plan.train_config["optimizer_args"] == ["weight_decay=0.05", "eps=1e-7"]
    # None in config_overrides removes them all.
    plan = _plan(image_dataset, tmp_path, optimizer="Prodigy", config_overrides={"optimizer_args": None})
    assert "optimizer_args" not in plan.train_config


@needs_submodule
@pytest.mark.parametrize("name", ["inverse_sqrt", "polynomial", "warmup_stable_decay", "cosine_with_restarts"])
def test_typed_schedulers_the_backend_knows_pass(image_dataset, tmp_path, name):
    plan = _plan(image_dataset, tmp_path, form_values={"lr_scheduler": name})
    assert plan.train_config["lr_scheduler"] == name


@needs_submodule
@pytest.mark.parametrize(
    "name, needs, given",
    [
        ("cosine_with_min_lr", "lr_scheduler_min_lr_ratio", {"lr_scheduler_min_lr_ratio": 0.1}),
        ("cosine_with_min_lr", "lr_scheduler_min_lr_ratio", {"lr_scheduler_args": ["min_lr=1e-6"]}),
        ("piecewise_constant", "step_rules", {"lr_scheduler_args": ["step_rules='1:500,0.1'"]}),
    ],
)
def test_schedulers_that_need_one_more_setting_ask_for_it(image_dataset, tmp_path, name, needs, given):
    """The backend sends cosine_with_min_lr no floor and piecewise_constant no
    step rules of its own, and both fail when training starts without them."""
    with pytest.raises(ValueError, match=needs):
        _plan(image_dataset, tmp_path, form_values={"lr_scheduler": name})
    plan = _plan(image_dataset, tmp_path, form_values={"lr_scheduler": name}, config_overrides=given)
    assert plan.train_config["lr_scheduler"] == name


@needs_submodule
@pytest.mark.parametrize(
    "name", ["reduce_lr_on_plateau", "greedy", "cosine_warmup_with_min_lr", "adafactor", "adafactor:0.001"]
)
def test_schedulers_the_backend_cannot_build_are_refused_by_name(image_dataset, tmp_path, name):
    with pytest.raises(ValueError, match="cannot be used here") as error:
        _plan(image_dataset, tmp_path, form_values={"lr_scheduler": name})
    assert name in str(error.value)


@needs_submodule
@pytest.mark.parametrize(
    "floor, conflict",
    [
        ({"lr_scheduler_min_lr_ratio": 0.1, "lr_scheduler_args": ["min_lr=1e-6"]}, "not both"),
        ({"lr_scheduler_args": ["min_lr_rate=0.1"]}, "not as min_lr_rate"),
    ],
)
def test_cosine_with_min_lr_takes_one_floor(image_dataset, tmp_path, floor, conflict):
    """transformers refuses both floors at once, and the backend already
    passes the ratio as min_lr_rate, so a second one is a duplicate."""
    with pytest.raises(ValueError, match=conflict):
        _plan(image_dataset, tmp_path, form_values={"lr_scheduler": "cosine_with_min_lr"}, config_overrides=floor)


@needs_submodule
def test_adafactor_in_relative_step_mode_skips_the_scheduler_check(image_dataset, tmp_path, capsys):
    """In relative-step mode the backend builds its own adafactor:<lr> schedule, so the picked one is not refused."""
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="AdaFactor",
        optimizer_recommended=False,
        form_values={"lr_scheduler": "adafactor", "lr_warmup_steps": 0.05},
    )
    assert "lr_warmup_steps" not in plan.train_config
    assert "relative-step mode" in capsys.readouterr().out
    # The early 1-click check agrees.
    route = registry.get("illustrious")
    common = dict(optimizer="AdaFactor", form_values={"lr_scheduler": "adafactor"}, sd_scripts_dir=SD_SCRIPTS)
    train.check_before_downloads(route, dict(optimizer_recommended=False, **common))
    # With the recommended settings the mode is off, so adafactor is refused as before.
    with pytest.raises(ValueError, match="cannot be used here"):
        _plan(image_dataset, tmp_path, optimizer="AdaFactor", form_values={"lr_scheduler": "adafactor"})
    with pytest.raises(ValueError, match="cannot be used here"):
        train.check_before_downloads(route, common)


def test_typed_scheduler_names_have_no_fallback_outside_the_list(monkeypatch):
    """A name the installed transformers knows but the list leaves out is still refused."""
    optimization = types.ModuleType("transformers.optimization")
    optimization.SchedulerType = [types.SimpleNamespace(value="something_new")]
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.optimization", optimization)
    with pytest.raises(ValueError, match="not a scheduler the backend knows"):
        train.check_scheduler("something_new")


@needs_submodule
def test_scheduler_typos_are_refused_unless_a_scheduler_class_is_named(image_dataset, tmp_path, capsys):
    with pytest.raises(ValueError, match="Did you mean cosine"):
        _plan(image_dataset, tmp_path, form_values={"lr_scheduler": "cosin"})
    # The backend uses the lr_scheduler_type class instead, and refuses warmup for a class without one.
    custom = {"lr_scheduler_type": "CosineAnnealingLR", "lr_scheduler_args": ["T_max=100"]}
    plan = _plan(
        image_dataset, tmp_path, form_values={"lr_scheduler": "cosin", "lr_warmup_steps": 0.05}, config_overrides=custom
    )
    assert plan.train_config["lr_scheduler_type"] == "CosineAnnealingLR"
    assert "lr_warmup_steps" not in plan.train_config  # the preset's 0.05 too
    assert "warmup is dropped" in capsys.readouterr().out
    # A warmup the user sets is theirs.
    plan = _plan(image_dataset, tmp_path, config_overrides={**custom, "lr_warmup_steps": 100})
    assert plan.train_config["lr_warmup_steps"] == 100
    # A schedule-free optimizer uses no scheduler, so its name is not checked.
    _plan(image_dataset, tmp_path, optimizer="AdamWScheduleFree", form_values={"lr_scheduler": "cosin"})


@needs_submodule
def test_warmup_stable_decay_without_decay_steps_gets_a_note(image_dataset, tmp_path, capsys):
    form = {"lr_scheduler": "warmup_stable_decay"}
    _plan(image_dataset, tmp_path, form_values=form)
    assert "lr_decay_steps" in capsys.readouterr().out
    _plan(image_dataset, tmp_path, form_values=form, config_overrides={"lr_decay_steps": 0.1})
    assert "lr_decay_steps" not in capsys.readouterr().out


@needs_submodule
def test_the_form_cycle_count_goes_only_to_cosine_with_restarts(image_dataset, tmp_path):
    """The backend passes half the count to warmup_stable_decay and cosine_with_min_lr,
    where the form's 3 makes the rate fall and rise again."""
    form = {"lr_scheduler": "warmup_stable_decay", "lr_scheduler_num_cycles": 3}
    for planner in (_plan, _sd15_plan):  # SD1.5's preset also sets 3
        plan = planner(image_dataset, tmp_path, form_values=form, config_overrides={"lr_decay_steps": 0.1})
        assert "lr_scheduler_num_cycles" not in plan.train_config
        assert plan.train_config["lr_decay_steps"] == 0.1
    plan = _plan(image_dataset, tmp_path, form_values={**form, "lr_scheduler": "cosine_with_restarts"})
    assert plan.train_config["lr_scheduler_num_cycles"] == 3
    # A count in config_overrides is always kept.
    plan = _plan(image_dataset, tmp_path, form_values=form, config_overrides={"lr_scheduler_num_cycles": 2})
    assert plan.train_config["lr_scheduler_num_cycles"] == 2


@needs_submodule
def test_adafactor_relative_step_mode_drops_warmup(image_dataset, tmp_path):
    """AdaFactor's default relative-step mode swaps in adafactor:<lr>, which refuses warmup
    like plain constant, after latent caching."""
    plan = _plan(image_dataset, tmp_path, optimizer="AdaFactor", optimizer_recommended=False)
    assert "lr_warmup_steps" not in plan.train_config
    # The recommended set has relative_step=False, so the normal scheduler runs.
    plan = _plan(image_dataset, tmp_path, optimizer="AdaFactor")
    assert "relative_step=False" in plan.train_config["optimizer_args"]
    assert plan.train_config["lr_warmup_steps"] == 0.05
    # A typed relative_step=False keeps the warmup too.
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="AdaFactor",
        optimizer_recommended=False,
        optimizer_args="relative_step=False",
    )
    assert plan.train_config["lr_warmup_steps"] == 0.05
    # warmup_init turns relative-step mode back on, so the warmup is still dropped.
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="AdaFactor",
        optimizer_recommended=False,
        optimizer_args="relative_step=False warmup_init=True",
    )
    assert "lr_warmup_steps" not in plan.train_config


@needs_submodule
def test_te_caching_requires_unet_only(image_dataset, tmp_path):
    """The backend checks this pair only after the dataset loads, so the plan refuses it first."""
    with pytest.raises(ValueError, match="network_train_unet_only"):
        _plan(image_dataset, tmp_path, config_overrides={"network_train_unet_only": False})
    with pytest.raises(ValueError, match="network_train_unet_only"):
        _plan(image_dataset, tmp_path, config_overrides={"network_train_unet_only": None})


@needs_submodule
def test_zero_te_lr_switches_to_unet_only(image_dataset, tmp_path):
    """In the backend, only network_train_unet_only freezes the text encoder. A rate of 0
    trains nothing and still takes the full time."""
    plan = _sd15_plan(image_dataset, tmp_path, config_overrides={"text_encoder_lr": 0})
    assert plan.train_config["network_train_unet_only"] is True
    assert "text_encoder_lr" not in plan.train_config
    # A rate above 0 keeps the text encoder training.
    plan = _sd15_plan(image_dataset, tmp_path, config_overrides={"text_encoder_lr": 1e-4})
    assert "network_train_unet_only" not in plan.train_config
    assert plan.train_config["text_encoder_lr"] == 1e-4
    # Prodigy's recommended set replaces the rate fields first, so 0 never reaches the check.
    plan = _sd15_plan(image_dataset, tmp_path, optimizer="Prodigy", form_values={"text_encoder_lr": 0})
    assert plan.train_config["text_encoder_lr"] == 1.0
    assert "network_train_unet_only" not in plan.train_config


@needs_submodule
def test_optimizer_args_are_read_the_way_the_backend_reads_them(image_dataset, tmp_path, capsys):
    """The backend reads each value with ast.literal_eval, so warmup_init=1.0 turns AdaFactor's own mode on."""
    plan = _plan(
        image_dataset,
        tmp_path,
        optimizer="AdaFactor",
        optimizer_recommended=False,
        optimizer_args="relative_step=False warmup_init=1.0",
        form_values={"lr_scheduler": "adafactor", "lr_warmup_steps": 0.05},
    )
    assert "lr_warmup_steps" not in plan.train_config
    assert "relative-step mode" in capsys.readouterr().out
    for bad, match in (("relative_step=false", "cannot read"), ("weight_decay=0.01=2", "exactly one '='")):
        with pytest.raises(ValueError, match=match):
            _plan(image_dataset, tmp_path, optimizer="AdamW", optimizer_args=bad)
        with pytest.raises(ValueError, match=match):
            train.check_before_downloads(ROUTE, dict(optimizer="AdamW", optimizer_args=bad, sd_scripts_dir=SD_SCRIPTS))


@needs_submodule
def test_a_string_of_scheduler_args_becomes_a_list(image_dataset, tmp_path):
    """The backend reads a string of lr_scheduler_args letter by letter and fails after caching."""
    plan = _plan(
        image_dataset,
        tmp_path,
        form_values={"lr_scheduler": "cosine_with_min_lr"},
        config_overrides={"lr_scheduler_args": "min_lr=1e-6"},
    )
    assert plan.train_config["lr_scheduler_args"] == ["min_lr=1e-6"]


@needs_submodule
def test_anima_preset_betas_go_only_to_its_own_optimizer(image_dataset, tmp_path):
    """The author's betas are written for AdamW8bit. CAME takes three betas."""
    plan = _anima_plan(image_dataset, tmp_path)
    assert plan.train_config["optimizer_args"] == ["betas=0.9,0.99"]
    assert plan.train_config["network_alpha"] == plan.train_config["network_dim"] == 32
    assert plan.train_config["sigmoid_scale"] == 1.3
    # The field adds to them, and config_overrides replaces one by name.
    plan = _anima_plan(image_dataset, tmp_path, optimizer_args="weight_decay=0.01")
    assert plan.train_config["optimizer_args"] == ["betas=0.9,0.99", "weight_decay=0.01"]
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"optimizer_args": ["betas=0.9,0.999"]})
    assert plan.train_config["optimizer_args"] == ["betas=0.9,0.999"]
    # Another optimizer runs without them.
    plan = _anima_plan(image_dataset, tmp_path, optimizer="CAME")
    assert plan.train_config["optimizer_args"] == ["weight_decay=0.04"]
    plan = _anima_plan(image_dataset, tmp_path, optimizer="AdamW")
    assert "optimizer_args" not in plan.train_config
    # The backend reads a plain name in any case, so a typed adamw8bit keeps them.
    plan = _anima_plan(image_dataset, tmp_path, optimizer="adamw8bit")
    assert plan.train_config["optimizer_args"] == ["betas=0.9,0.99"]
    preset = config.load_preset(registry.get("anima").preset)
    assert train.preset_optimizer_args(preset, "ADAMW8BIT") == ["betas=0.9,0.99"]
    assert train.preset_optimizer_args(preset, "") == ["betas=0.9,0.99"]
    assert train.preset_optimizer_args(preset, "pytorch_optimizer.CAME") == []

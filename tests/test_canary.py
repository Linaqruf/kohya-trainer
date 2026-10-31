"""The canary's network-free stages and its failure path (scripts/canary_smoke.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import huggingface_hub
import pytest
import yaml
from huggingface_hub.errors import GatedRepoError

from kohya_trainer import config, dataset, registry, tagging, train

from conftest import ROOT, SD_SCRIPTS, needs_submodule


def _canary():
    spec = importlib.util.spec_from_file_location("canary_smoke", ROOT / "scripts" / "canary_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANARY = _canary()


def test_png_fixture_is_a_real_png():
    image = CANARY.png_1px()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in image and b"IDAT" in image and b"IEND" in image


def test_synthetic_dataset_is_counted_by_the_trainer_glob(tmp_path):
    made = CANARY.make_dataset(tmp_path / "ds")
    summary = dataset.summarize(made)
    assert summary == {"images": 4, "captioned": 4, "uncaptioned": 0, "nested_images": 0}


def test_entrypoint_modules_cover_every_route():
    modules = CANARY.entrypoint_modules()
    assert modules == sorted(modules)
    assert len(modules) == len(set(modules))
    for route in registry.ROUTES.values():
        assert route.entrypoint.removesuffix(".py") in modules


def test_curated_artifacts_cover_registry_and_taggers():
    artifacts = CANARY.curated_artifacts()
    assert len(artifacts) == len(set(artifacts))  # deduplicated
    for route in registry.ROUTES.values():
        for spec in (*route.base_models, route.default_vae, route.default_text_encoder):
            if spec is not None:
                assert (spec.repo_id, spec.filename) in artifacts
                if spec.int8_filename:
                    assert (spec.repo_id, spec.int8_filename) in artifacts
        for spec in route.runtime_files:
            assert spec in artifacts
    assert ("Comfy-Org/MOSS-Image", "diffusion_models/moss_image_base_int8_convrot.safetensors") in artifacts
    assert ("OpenMOSS-Team/MOSS-Image-Base", "tokenizer/tokenizer.json") in artifacts
    for repo_id in tagging.TAGGERS:
        assert (repo_id, None) in artifacts


def test_backend_files_cover_preset_network_modules():
    presets = sorted((ROOT / "src" / "kohya_trainer" / "presets").glob("*.toml"))
    assert presets
    named = {config.load_preset(path.name).get("network_module") for path in presets} - {None}
    assert {"networks.lora_anima", "networks.lora_generic"} <= named
    assert CANARY.preset_network_modules() == sorted(named)
    files = CANARY.backend_files()
    assert len(files) == len(set(files))
    for module in named | set(train.NETWORK_MODULES.values()):
        assert Path(module.replace(".", "/") + ".py") in files


def _canary_workflow() -> dict:
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "canary.yml").read_text(encoding="utf-8"))
    # YAML 1.1 reads a bare `on` key as the boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def test_canary_runs_on_pull_requests_that_touch_the_backend():
    triggers = _canary_workflow()["on"]
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    paths = triggers["pull_request"]["paths"]
    # The submodule pointer changes as this exact path. A "/**" glob misses it.
    for path in (
        "external/sd-scripts",
        "pyproject.toml",
        "src/kohya_trainer/pins.py",
        "src/kohya_trainer/registry.py",
        "src/kohya_trainer/environment.py",
        "scripts/canary_smoke.py",
        ".github/workflows/canary.yml",
    ):
        assert path in paths


def test_canary_files_issues_only_for_scheduled_runs():
    workflow = _canary_workflow()
    # The check job runs a pull request's own code, so it gets no write access.
    assert workflow["permissions"] == {"contents": "read"}
    smoke_steps = workflow["jobs"]["colab-drift"]["steps"]
    assert not [step for step in smoke_steps if "gh issue" in step.get("run", "")]
    report = workflow["jobs"]["report-drift"]
    assert report["needs"] == "colab-drift"
    assert report["if"] == "failure() && github.event_name == 'schedule'"
    assert report["permissions"] == {"issues": "write"}
    assert not [step for step in report["steps"] if "uses" in step], "the report job must not check out code"
    assert any("gh issue create" in step.get("run", "") for step in report["steps"])


def test_pull_request_runs_cannot_cancel_the_weekly_run():
    concurrency = _canary_workflow()["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    assert "github.event.pull_request.number" in concurrency["group"]


@needs_submodule
def test_backend_files_exist_at_the_pin():
    CANARY.check_backend_files()
    for relative in CANARY.backend_files():
        assert (SD_SCRIPTS / relative).is_file()


@needs_submodule
def test_dry_run_plans_cover_every_route_and_base(tmp_path):
    plans = CANARY.dry_run_plans(tmp_path)
    expected = {(route.key, base.key) for route in registry.ROUTES.values() for base in route.base_models}
    assert {(plan.route_key, plan.base_key) for plan in plans} == expected
    for plan in plans:
        assert plan.config_file.is_file()
        assert plan.dataset_config_file.is_file()
        assert plan.command[0] == "accelerate"
    # The ConvRot flags a pasted base gets pass the key check on their route.
    convrot = [plan for plan in plans if plan.train_config.get("convrot_int8")]
    assert {plan.route_key for plan in convrot} == {key for key, r in registry.ROUTES.items() if r.convrot_int8}
    assert all(plan.train_config.get("convrot_int8_text_encoder") for plan in convrot)


@needs_submodule
def test_runtime_files_are_the_ones_the_trainer_fetches():
    """The MOSS-Image trainer loads its tokenizer from the Hub when it starts."""
    utils = (SD_SCRIPTS / "library" / "moss_image_utils.py").read_text(encoding="utf-8")
    files = registry.get("moss_image").runtime_files
    for repo_id, filename in files:
        assert f'TOKENIZER_ID = "{repo_id}"' in utils
        folder, _, name = filename.partition("/")
        assert f'TOKENIZER_SUBFOLDER = "{folder}"' in utils
        assert name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _fake_stack(monkeypatch, schedulers, modules):
    """Stand-ins for transformers.optimization and the optimizer packages."""
    import enum
    import sys
    import types

    optimization = types.ModuleType("transformers.optimization")
    optimization.SchedulerType = enum.Enum("SchedulerType", {name.upper(): name for name in schedulers})
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.optimization", optimization)
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, type(attribute, (), {}))
        monkeypatch.setitem(sys.modules, name, module)


_OPTIMIZER_MODULES = {"pytorch_optimizer": ["CAME"], "prodigyplus": ["ProdigyPlusScheduleFree"]}


def test_training_names_pass_with_a_stack_that_has_them(monkeypatch):
    _fake_stack(monkeypatch, train.TRANSFORMERS_SCHEDULERS, _OPTIMIZER_MODULES)
    CANARY.check_training_names()


def test_training_names_fail_on_a_scheduler_transformers_lacks(monkeypatch):
    _fake_stack(monkeypatch, [n for n in train.TRANSFORMERS_SCHEDULERS if n != "cosine"], _OPTIMIZER_MODULES)
    with pytest.raises(RuntimeError, match="cosine"):
        CANARY.check_training_names()


def test_training_names_fail_on_a_listed_optimizer_that_is_gone(monkeypatch):
    _fake_stack(monkeypatch, train.TRANSFORMERS_SCHEDULERS, {**_OPTIMIZER_MODULES, "pytorch_optimizer": ["Lamb"]})
    with pytest.raises(RuntimeError, match="pytorch_optimizer.CAME"):
        CANARY.check_training_names()


# --- the failure path: the canary must report a missing Hub file ---


class _FakeApi:
    """HfApi stand-in with scripted verdicts per repo_id."""

    def __init__(self, behavior):
        self.behavior = behavior  # repo_id -> True | False | Exception instance

    def _answer(self, repo_id):
        verdict = self.behavior.get(repo_id, True)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    def file_exists(self, repo_id, filename):
        return self._answer(repo_id)

    def repo_exists(self, repo_id):
        return self._answer(repo_id)


def test_check_artifacts_reports_missing(monkeypatch):
    gone = registry.ROUTES["sd15"].base_models[0].repo_id
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _FakeApi({gone: False}))
    missing, inconclusive = CANARY.check_artifacts()
    assert missing and all(label.startswith(gone) for label in missing)
    assert inconclusive == []


class _FakeGated(GatedRepoError):
    """A GatedRepoError that skips its constructor, whose signature differs across huggingface_hub versions."""

    def __init__(self, message):
        Exception.__init__(self, message)


def test_exists_treats_gated_as_present():
    api = _FakeApi({"circlestone-labs/Anima": _FakeGated("license wall")})
    assert CANARY._exists(api, "circlestone-labs/Anima", "some/file.safetensors") is True


def test_exists_returns_inconclusive_on_persistent_network_trouble(monkeypatch):
    monkeypatch.setattr(CANARY.time, "sleep", lambda seconds: None)
    api = _FakeApi({"down/repo": ConnectionError("no route to Hub")})
    assert CANARY._exists(api, "down/repo", None) is None


@pytest.mark.parametrize("filename", ["file.safetensors", None])
def test_exists_definitive_verdicts_pass_through(filename):
    assert CANARY._exists(_FakeApi({"a/b": True}), "a/b", filename) is True
    assert CANARY._exists(_FakeApi({"a/b": False}), "a/b", filename) is False

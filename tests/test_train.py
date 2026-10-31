from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import math
import re
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from huggingface_hub import CommitOperationAdd

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from kohya_trainer import config, models, registry, train
from kohya_trainer import plan as plan_mod

from conftest import ROUTE, SD_SCRIPTS, _anima_plan, _fake_plan, _plan, needs_submodule


@needs_submodule
def test_build_plan_writes_configs_and_command(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path)
    assert plan.config_file.is_file()
    assert plan.dataset_config_file.is_file()
    assert plan.sample_prompts_file.is_file()
    assert plan.command[:2] == ["accelerate", "launch"]
    assert plan.command[3].endswith("sdxl_train_network.py")
    written = tomllib.loads(plan.config_file.read_text(encoding="utf-8"))
    assert written == plan.train_config
    assert written["sample_prompts"] == str(plan.sample_prompts_file)


@needs_submodule
def test_plan_estimates_run_length(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path)  # 3 images, auto-repeats 20, batch 2
    assert plan.estimated_steps_per_epoch == math.ceil(3 * 20 / 2)
    assert plan.estimated_total_steps == plan.estimated_steps_per_epoch * plan.train_config["max_train_epochs"]


@needs_submodule
def test_planning_never_deletes_text_caches(image_dataset, tmp_path):
    """Only execute removes old caches. The advanced notebook plans first."""
    cache = image_dataset / "a_te_outputs.npz"
    cache.write_bytes(b"te")
    _plan(image_dataset, tmp_path, trigger_word="zkx")
    assert cache.is_file()


@needs_submodule
def test_low_ram_machines_get_the_full_precision_lowram_pair(image_dataset, tmp_path, monkeypatch):
    """Free Colab (12.7 GB RAM) cannot load fp32 SDXL in CPU memory, and lowram
    alone puts the fp32 copy on the GPU. The backend loads the smaller type only with full_*."""
    monkeypatch.setattr(train.environment, "total_ram_gb", lambda: 12.7)
    plan = _plan(image_dataset, tmp_path)
    assert plan.train_config["lowram"] is True
    assert plan.train_config["full_fp16"] is True  # illustrious preset trains fp16
    # Anima loads its DiT to the GPU in bf16, and full_bf16 changes how it trains.
    plan = _anima_plan(image_dataset, tmp_path)
    assert "full_bf16" not in plan.train_config
    assert "lowram" not in plan.train_config
    # The user's own None wins.
    plan = _plan(image_dataset, tmp_path, config_overrides={"lowram": None, "full_fp16": None})
    assert "lowram" not in plan.train_config and "full_fp16" not in plan.train_config
    monkeypatch.setattr(train.environment, "total_ram_gb", lambda: 64.0)
    plan = _plan(image_dataset, tmp_path)
    assert "lowram" not in plan.train_config and "full_fp16" not in plan.train_config


def test_execute_streams_output_and_carries_the_tail(tmp_path, capsys):
    """Colab cells do not show a child process's stdout, so execute passes it on
    and puts its tail in the error."""
    boom = [sys.executable, "-c", "print('backend says hi'); raise SystemExit(3)"]
    with pytest.raises(RuntimeError) as error:
        train.execute(_fake_plan(boom, tmp_path), sd_scripts_dir=tmp_path)
    assert "backend says hi" in capsys.readouterr().out  # streamed live
    assert "backend says hi" in str(error.value)  # carried in the traceback
    assert "code 3" in str(error.value)
    ok = [sys.executable, "-c", "print('all good')"]
    # A notebook cell prints a returned value under the run, so nothing is returned.
    assert train.execute(_fake_plan(ok, tmp_path), sd_scripts_dir=tmp_path) is None
    out = capsys.readouterr().out
    assert "all good" in out and "Done." in out


@needs_submodule
def test_upload_repo_wires_the_backend_native_async_upload(image_dataset, tmp_path):
    """The backend uploads each saved file. The token stays in HF_TOKEN, never in the config."""
    plan = _plan(image_dataset, tmp_path, upload_repo="user/my-lora")
    assert plan.train_config["huggingface_repo_id"] == "user/my-lora"
    assert plan.train_config["async_upload"] is True
    assert "huggingface_repo_visibility" not in plan.train_config  # default = private
    assert "huggingface_token" not in plan.train_config
    plan = _plan(image_dataset, tmp_path, upload_repo="user/my-lora", upload_public=True)
    assert plan.train_config["huggingface_repo_visibility"] == "public"
    plan = _plan(image_dataset, tmp_path)  # no repo, no upload keys
    assert "huggingface_repo_id" not in plan.train_config
    assert "async_upload" not in plan.train_config
    assert "huggingface_path_in_repo" not in plan.train_config


@needs_submodule
def test_upload_goes_to_a_folder_named_after_the_project(image_dataset, tmp_path, capsys):
    plan = _plan(image_dataset, tmp_path, upload_repo="user/loras")
    assert plan.train_config["huggingface_path_in_repo"] == "test-lora"
    # The default is printed when it applies.
    assert "user/loras, in the folder test-lora/" in capsys.readouterr().out
    plan = _plan(image_dataset, tmp_path, upload_repo="user/loras", config_overrides={"huggingface_path_in_repo": "v2"})
    assert plan.train_config["huggingface_path_in_repo"] == "v2"
    capsys.readouterr()
    plan = _plan(image_dataset, tmp_path, upload_repo="user/loras", project_name=".cache")
    assert plan.train_config["huggingface_path_in_repo"] == ""
    assert "repo root" in capsys.readouterr().out
    # A None override must not drop the key: with no path, every upload fails.
    plan = _plan(image_dataset, tmp_path, upload_repo="user/loras", config_overrides={"huggingface_path_in_repo": None})
    assert plan.train_config["huggingface_path_in_repo"] == ""
    # The note names the repo the backend will use.
    capsys.readouterr()
    _plan(image_dataset, tmp_path, upload_repo="user/loras", config_overrides={"huggingface_repo_id": "other/repo"})
    assert "goes to other/repo" in capsys.readouterr().out


def _backend_upload_module(monkeypatch):
    """Load the backend's library/huggingface_util.py with library.utils stubbed, so without torch.

    fire_in_thread runs inline, so the test sees the upload."""
    library = types.ModuleType("library")
    library.__path__ = []
    utils = types.ModuleType("library.utils")
    utils.fire_in_thread = lambda f, *args, **kwargs: f(*args, **kwargs)
    utils.setup_logging = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "library", library)
    monkeypatch.setitem(sys.modules, "library.utils", utils)
    spec = importlib.util.spec_from_file_location(
        "backend_huggingface_util", SD_SCRIPTS / "library" / "huggingface_util.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHub:
    """Offline HfApi that records where each file lands, through the Hub library's own path checks."""

    landed: ClassVar[list[str]] = []

    def __init__(self, token=None):
        assert token is None  # the token stays in HF_TOKEN, never in the config

    def repo_info(self, **kwargs):
        return object()

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type=None):
        operation = CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)
        self.landed.append(operation.path_in_repo)


def _backend_args(plan) -> argparse.Namespace:
    # The backend's defaults for the upload arguments, then the written config.
    args = argparse.Namespace(
        huggingface_repo_id=None,
        huggingface_repo_type=None,
        huggingface_path_in_repo=None,
        huggingface_token=None,
        huggingface_repo_visibility=None,
        async_upload=False,
    )
    vars(args).update(tomllib.loads(plan.config_file.read_text(encoding="utf-8")))
    return args


@needs_submodule
@pytest.mark.parametrize("project", ["my_lora", "my lora", "mychar v2", ".git", ".cache"])
def test_backend_upload_lands_with_the_planned_path(image_dataset, tmp_path, monkeypatch, project):
    """Call the backend's upload() as train_network.py does, for an epoch file and the final file."""
    hub = _backend_upload_module(monkeypatch)
    monkeypatch.setattr(hub, "HfApi", _FakeHub)
    monkeypatch.setattr(_FakeHub, "landed", [])
    plan = _plan(image_dataset, tmp_path, upload_repo="user/loras", project_name=project)
    args = _backend_args(plan)
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"x")
    names = [f"{project}-000002.safetensors", f"{project}.safetensors"]
    for name in names:
        hub.upload(args, str(lora), "/" + name)
    folder = args.huggingface_path_in_repo
    assert _FakeHub.landed == [f"{folder}/{name}" if folder else name for name in names]
    if project == "my_lora":
        assert _FakeHub.landed == ["my_lora/my_lora-000002.safetensors", "my_lora/my_lora.safetensors"]


@needs_submodule
def test_backend_upload_without_a_path_never_lands(tmp_path, monkeypatch, caplog):
    """Without huggingface_path_in_repo the backend passes None, the Hub library
    raises, and the backend only logs it."""
    hub = _backend_upload_module(monkeypatch)
    monkeypatch.setattr(hub, "HfApi", _FakeHub)
    monkeypatch.setattr(_FakeHub, "landed", [])
    lora = tmp_path / "lora.safetensors"
    lora.write_bytes(b"x")
    args = argparse.Namespace(
        huggingface_repo_id="user/loras",
        huggingface_repo_type=None,
        huggingface_path_in_repo=None,
        huggingface_token=None,
        huggingface_repo_visibility=None,
        async_upload=False,
    )
    with caplog.at_level(logging.ERROR):
        hub.upload(args, str(lora), "/my_lora-000002.safetensors")
    assert _FakeHub.landed == []
    # The cause must be the None path, not another error the backend's catch-all hides.
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "failed to upload" in logged and "startswith" in logged, logged


@needs_submodule
def test_llm_adapter_opt_in_gets_reduced_adapter_lr(image_dataset, tmp_path):
    """The LLM adapter trains more stably at a lower learning rate (upstream Anima doc)."""
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"network_args": ["train_llm_adapter=True"]})
    assert "network_reg_lrs=.*llm_adapter.*=5e-5" in plan.train_config["network_args"]
    # An explicit per-module LR wins over the injected default.
    plan = _anima_plan(
        image_dataset,
        tmp_path,
        config_overrides={"network_args": ["train_llm_adapter=True", "network_reg_lrs=.*llm_adapter.*=1e-5"]},
    )
    args = plan.train_config["network_args"]
    assert args.count("network_reg_lrs=.*llm_adapter.*=1e-5") == 1
    assert "network_reg_lrs=.*llm_adapter.*=5e-5" not in args
    # No opt-in, no injection.
    plan = _anima_plan(image_dataset, tmp_path)
    assert "network_args" not in plan.train_config


@needs_submodule
def test_adapter_rate_and_note_follow_the_route(image_dataset, tmp_path, capsys):
    """The switch names, the module pattern and the note's words are route data."""
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"network_args": ["train_llm_adapter=TRUE"]})
    assert "network_reg_lrs=.*llm_adapter.*=5e-5" in plan.train_config["network_args"]
    assert "the LLM adapter's learning rate is set to 5e-5" in capsys.readouterr().out
    # The backend splits at the first = and compares the rest with "true" as it
    # is, so "1", a space after = or a space in the name all read as off.
    for arg in ("train_llm_adapter=1", "train_llm_adapter= True", "train_llm_adapter =True"):
        with pytest.raises(ValueError, match='reads as off.*Write "train_llm_adapter=True" without spaces'):
            _anima_plan(image_dataset, tmp_path, config_overrides={"network_args": [arg]})
    # Words that mean off read as off, as meant.
    for arg in ("train_llm_adapter=False", "train_llm_adapter=0", "train_llm_adapter=no"):
        plan = _anima_plan(image_dataset, tmp_path, config_overrides={"network_args": [arg]})
        assert not any("network_reg_lrs" in entry for entry in plan.train_config["network_args"]), arg
    # A route with no adapter refuses the switch, however it is spelled.
    for arg in ("train_llm_adapter=True", "train_llm_adapter=1"):
        with pytest.raises(ValueError, match="trains an adapter that SDXL does not have"):
            train.check_before_downloads(registry.get("sdxl"), {"config_overrides": {"network_args": [arg]}})
    embedder = dataclasses.replace(
        registry.get("anima"),
        adapter_args=("train_text_embedder", "train_llm_adapter"),
        adapter_pattern=".*text_embedder.*",
        adapter_name="the text embedder",
    )
    text_encoder = tmp_path / "te.safetensors"
    text_encoder.touch()
    plan = train.build_plan(
        embedder,
        embedder.base(),
        checkpoint_path=Path("/models/dit.safetensors"),
        vae_path=None,
        text_encoder_path=text_encoder,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws-embedder",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
        config_overrides={"network_module": "networks.lora_generic", "network_args": ["train_text_embedder=True"]},
    )
    assert "network_reg_lrs=.*text_embedder.*=5e-5" in plan.train_config["network_args"]
    assert "train_text_embedder is on, so the text embedder's learning rate" in capsys.readouterr().out


@needs_submodule
def test_an_adapter_switch_the_network_ignores_is_refused(image_dataset, tmp_path):
    """lora_anima reads only train_llm_adapter, so with the MOSS-Image name the
    run trains without the adapter and says nothing."""
    for arg in ("train_text_embedder=True", "train_text_embedder=1"):  # the name first, then the spelling
        with pytest.raises(ValueError, match=r'does nothing with networks.lora_anima.*Use "train_llm_adapter=True"'):
            _anima_plan(image_dataset, tmp_path, config_overrides={"network_args": [arg]})
    # The same check runs before any download.
    with pytest.raises(ValueError, match="the LLM adapter does not train"):
        train.check_before_downloads(
            registry.get("anima"),
            {"network_type": "loha", "config_overrides": {"network_args": ["train_text_embedder=true"]}},
        )
    # A switch the module reads passes.
    train.check_before_downloads(
        registry.get("anima"),
        {"network_type": "loha", "config_overrides": {"network_args": ["train_llm_adapter=True"]}},
    )


@needs_submodule
def test_adapter_switches_match_what_each_network_module_reads():
    for module, names in train.ADAPTER_ARGS_READ.items():
        source = (SD_SCRIPTS / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert set(re.findall(r'kwargs\.get\(\s*"(train_(?:llm_adapter|text_embedder))"', source)) == set(names), module
    for route in registry.ROUTES.values():
        if not route.adapter_args:
            continue
        preset = config.load_preset(route.preset)
        for network_type in route.network_types:
            module = train.network_module_that_runs(network_type, {}, preset)
            assert set(train.ADAPTER_ARGS_READ[module]) & set(route.adapter_args), (route.key, module)


@needs_submodule
def test_quick_train_dry_run_end_to_end(image_dataset, tmp_path, fake_download, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    plan = train.quick_train(
        model="illustrious",
        dataset=image_dataset,
        project_name="dry",
        workspace=tmp_path / "runs",
        auto_tag=False,
        dry_run=True,
        log_to_wandb=True,  # the 1-click field must reach build_plan intact
    )
    assert plan.base_key == ROUTE.default_base
    assert plan.train_config["pretrained_model_name_or_path"] == str(fake_download)
    assert plan.train_config["log_with"] == "wandb"
    assert plan.config_file.is_file()
    assert (tmp_path / "runs" / "dry" / "output").is_dir()


@pytest.mark.parametrize("name", ["", "   ", "a/b", "a\\b", "/abs", ".", ".."])
@needs_submodule
def test_build_plan_refuses_a_project_name_that_cannot_be_a_folder(image_dataset, tmp_path, name):
    with pytest.raises(ValueError, match="project_name") as error:
        _plan(image_dataset, tmp_path, project_name=name)
    assert "my-lora" in str(error.value)
    assert not (tmp_path / "ws").exists()  # refused before any folder is made


@needs_submodule
def test_ordinary_project_names_still_plan(image_dataset, tmp_path):
    for name in ("my lora", "my-lora_v2", "char.v1"):
        plan = _plan(image_dataset, tmp_path, project_name=name)
        assert plan.train_config["output_name"] == name


def test_quick_train_refuses_a_bad_project_name_before_downloading(tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("downloaded before checking the project name")

    monkeypatch.setattr(train.models, "prepare_base", no_download)
    monkeypatch.setattr(train.dataset_mod, "prepare_dataset", no_download)
    with pytest.raises(ValueError, match="project_name"):
        train.quick_train(model="sdxl", dataset=tmp_path, project_name="a/b", workspace=tmp_path / "runs", dry_run=True)
    assert not (tmp_path / "runs").exists()


@needs_submodule
def test_network_type_selects_lycoris_module(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, network_type="loha")
    assert plan.train_config["network_module"] == "networks.loha"
    plan = _plan(image_dataset, tmp_path, network_type="lokr")
    assert plan.train_config["network_module"] == "networks.lokr"
    plan = _plan(image_dataset, tmp_path)
    assert plan.train_config["network_module"] == "networks.lora"
    with pytest.raises(ValueError, match="unknown network_type"):
        _plan(image_dataset, tmp_path, network_type="ia3")


@needs_submodule
def test_lycoris_refused_on_routes_without_arch_support(image_dataset, tmp_path):
    """detect_arch_config in networks/network_base.py raises for SD1.5, so the plan must refuse first."""
    sd15 = registry.get("sd15")
    with pytest.raises(ValueError, match="supports network types"):
        train.build_plan(
            sd15,
            sd15.base(),
            checkpoint_path=Path("/models/base.safetensors"),
            vae_path=None,
            dataset_dir=image_dataset,
            workspace=tmp_path / "ws15",
            project_name="test-sd15",
            network_type="loha",
            sd_scripts_dir=SD_SCRIPTS,
        )


def test_route_network_types_are_known():
    for route in registry.ROUTES.values():
        assert set(route.network_types) <= set(train.NETWORK_MODULES)
        assert route.network_types[0] == "lora"  # the notebook form's default


def test_locon_is_unet_only():
    assert "locon" in registry.get("sd15").network_types
    assert "locon" in registry.get("sdxl").network_types
    assert "locon" not in registry.get("anima").network_types  # DiT has no Conv2d


@needs_submodule
def test_lycoris_types_get_capacity_defaults(image_dataset, tmp_path):
    """LoHa at LoRA-sized dims overfits, and LoKr below full-matrix mode learns too
    little. The form's dim and alpha apply only to lora and locon."""
    plan = _plan(image_dataset, tmp_path, network_type="loha", network_dim=32, network_alpha=16)
    assert plan.train_config["network_dim"] == 16
    assert plan.train_config["network_alpha"] == 8
    plan = _plan(image_dataset, tmp_path, network_type="lokr")
    assert plan.train_config["network_dim"] == 10000
    assert plan.train_config["network_alpha"] == 1
    assert "factor=8" in plan.train_config["network_args"]
    # config_overrides still wins: its values replace the defaults.
    plan = _plan(
        image_dataset,
        tmp_path,
        network_type="loha",
        config_overrides={"network_dim": 8, "network_alpha": 4},
    )
    assert plan.train_config["network_dim"] == 8
    assert plan.train_config["network_alpha"] == 4
    plan = _plan(image_dataset, tmp_path, network_type="lokr", config_overrides={"network_args": ["factor=4"]})
    args = plan.train_config["network_args"]
    assert "factor=4" in args and "factor=8" not in args


@needs_submodule
def test_lokr_note_says_how_to_change_its_size_and_expects_the_backend_warning(image_dataset, tmp_path, capsys):
    _plan(image_dataset, tmp_path, network_type="lokr")
    out = capsys.readouterr().out
    assert 'add "network_args": ["factor=N"] to config_overrides' in out
    assert "full matrix mode once for each layer" in out
    _plan(image_dataset, tmp_path, network_type="loha")
    out = capsys.readouterr().out
    assert "add network_dim and network_alpha to config_overrides" in out
    assert "full matrix" not in out


@needs_submodule
def test_conv_form_fields_replace_the_locon_defaults(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, network_type="locon", conv_dim=16, conv_alpha=8)
    args = plan.train_config["network_args"]
    assert "conv_dim=16" in args and "conv_dim=8" not in args
    assert "conv_alpha=8" in args and "conv_alpha=4" not in args
    # Other types ignore the conv fields.
    plan = _plan(image_dataset, tmp_path, conv_dim=16)
    assert "network_args" not in plan.train_config


@needs_submodule
def test_lycoris_batch_hint_prints_only_on_a_detected_t4(image_dataset, tmp_path, capsys, monkeypatch):
    """The memory hint prints at plan time, before latent caching, and only for a batch above 1."""
    monkeypatch.setattr(train.environment, "gpu_name", lambda: "Tesla T4")
    _plan(image_dataset, tmp_path, network_type="loha")  # batch 2
    out = capsys.readouterr().out
    assert "train_batch_size=1" in out
    assert "free" not in out
    _plan(image_dataset, tmp_path, network_type="loha", train_batch_size=1)
    assert "train_batch_size=1" not in capsys.readouterr().out
    _anima_plan(image_dataset, tmp_path, network_type="loha")  # any route, keyed on the GPU
    assert "train_batch_size=1" in capsys.readouterr().out
    for detected in (lambda: "NVIDIA L4", lambda: None):  # an L4, or no GPU at all
        monkeypatch.setattr(train.environment, "gpu_name", detected)
        _plan(image_dataset, tmp_path, network_type="loha")
        assert "train_batch_size=1" not in capsys.readouterr().out


@needs_submodule
def test_form_dims_apply_to_lora_and_lose_to_config_overrides(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, network_dim=64, network_alpha=32)
    assert plan.train_config["network_dim"] == 64
    assert plan.train_config["network_alpha"] == 32
    plan = _plan(image_dataset, tmp_path, network_dim=64, config_overrides={"network_dim": 8})
    assert plan.train_config["network_dim"] == 8


@needs_submodule
def test_locon_adds_the_conv_pair_and_keeps_caller_dims(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, network_type="locon", network_dim=24)
    assert plan.train_config["network_module"] == "networks.lora"
    assert "conv_dim=8" in plan.train_config["network_args"]
    assert "conv_alpha=4" in plan.train_config["network_args"]
    assert plan.train_config["network_dim"] == 24  # form dims apply to LoCon
    plan = _plan(image_dataset, tmp_path, network_type="locon", config_overrides={"network_args": ["conv_dim=16"]})
    args = plan.train_config["network_args"]
    assert "conv_dim=16" in args and "conv_dim=8" not in args
    assert "conv_alpha=4" in args


@needs_submodule
def test_locon_refused_on_dit_routes(image_dataset, tmp_path):
    with pytest.raises(ValueError, match="supports network types"):
        _anima_plan(image_dataset, tmp_path, network_type="locon")


@needs_submodule
def test_anima_route_threads_text_encoder(image_dataset, tmp_path):
    anima = registry.get("anima")
    plan = train.build_plan(
        anima,
        anima.base(),
        checkpoint_path=Path("/models/anima-dit.safetensors"),
        vae_path=Path("/models/qwen_image_vae.safetensors"),
        text_encoder_path=Path("/models/qwen3.safetensors"),
        dataset_dir=image_dataset,
        workspace=tmp_path / "wsa",
        project_name="test-anima",
        sd_scripts_dir=SD_SCRIPTS,
    )
    assert plan.train_config["qwen3"] == str(Path("/models/qwen3.safetensors"))
    assert plan.train_config["network_module"] == "networks.lora_anima"
    assert plan.train_config["mixed_precision"] == "bf16"
    assert plan.train_config["timestep_sampling"] == "sigmoid"
    assert plan.command[3].endswith("anima_train_network.py")
    # no-op / SD-only keys must not leak into the anima config
    for absent in ("clip_skip", "min_snr_gamma", "no_half_vae", "sample_sampler", "v_parameterization"):
        assert absent not in plan.train_config, absent


@needs_submodule
def test_anima_route_requires_text_encoder(image_dataset, tmp_path):
    anima = registry.get("anima")
    with pytest.raises(ValueError, match="text-encoder artifact"):
        train.build_plan(
            anima,
            anima.base(),
            checkpoint_path=Path("/models/anima-dit.safetensors"),
            vae_path=Path("/models/qwen_image_vae.safetensors"),
            dataset_dir=image_dataset,
            workspace=tmp_path / "wsa",
            project_name="test-anima",
            sd_scripts_dir=SD_SCRIPTS,
        )


def test_custom_base_spec_parsing():
    base = models.custom_base("user/repo:file.safetensors")
    assert base.repo_id == "user/repo"
    assert base.filename == "file.safetensors"
    assert models.custom_base("user/repo").filename is None


@needs_submodule
def test_pointless_caching_override_gets_a_note(image_dataset, tmp_path, capsys):
    caching_off = {"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None}
    _plan(image_dataset, tmp_path, config_overrides=dict(caching_off))
    assert "save GPU memory" in capsys.readouterr().out
    # A trigger does not need caching off.
    _plan(image_dataset, tmp_path, trigger_word="zkx", config_overrides=dict(caching_off))
    assert "save GPU memory" in capsys.readouterr().out
    # Runs that need caching off get no note.
    _plan(image_dataset, tmp_path, config_overrides=dict(caching_off), dataset_overrides={"shuffle_caption": True})
    assert "save GPU memory" not in capsys.readouterr().out


@needs_submodule
def test_network_weights_warm_start(image_dataset, tmp_path):
    """dim and alpha come from the LoRA file (dim_from_weights), so they always match its shapes."""
    lora_file = tmp_path / "previous.safetensors"
    lora_file.write_bytes(b"00")
    plan = _plan(image_dataset, tmp_path, network_weights=str(lora_file))
    assert plan.train_config["network_weights"] == str(lora_file.resolve())
    assert plan.train_config["dim_from_weights"] is True
    plan = _plan(image_dataset, tmp_path)
    assert "network_weights" not in plan.train_config
    assert "dim_from_weights" not in plan.train_config
    with pytest.raises(FileNotFoundError, match="network_weights"):
        _plan(image_dataset, tmp_path, network_weights=str(tmp_path / "missing.safetensors"))


@needs_submodule
def test_lokr_warm_start_warns_about_factor(image_dataset, tmp_path, capsys):
    """dim_from_weights does not read LoKr's factor, so the plan says it must match."""
    lora_file = tmp_path / "previous-lokr.safetensors"
    lora_file.write_bytes(b"00")
    _plan(image_dataset, tmp_path, network_type="lokr", network_weights=str(lora_file))
    assert "factor=8" in capsys.readouterr().out


@needs_submodule
def test_gradient_accumulation_scales_the_estimate(image_dataset, tmp_path):
    """A step is one optimizer step, so the estimate divides by the accumulation."""
    plan = _plan(image_dataset, tmp_path, config_overrides={"gradient_accumulation_steps": 2})
    assert plan.train_config["gradient_accumulation_steps"] == 2
    assert plan.estimated_steps_per_epoch == math.ceil(3 * 20 / (2 * 2))


@needs_submodule
def test_save_every_zero_saves_only_the_final_file(image_dataset, tmp_path, capsys):
    """The backend computes (epoch + 1) % save_every_n_epochs, so a 0 stops the run after one epoch."""
    plan = _plan(image_dataset, tmp_path, form_values={"save_every_n_epochs": 0, "sample_every_n_epochs": 0})
    assert "save_every_n_epochs" not in plan.train_config
    assert "sample_every_n_epochs" not in plan.train_config
    assert "only the final LoRA is saved" in capsys.readouterr().out
    plan = _plan(image_dataset, tmp_path, form_values={"save_every_n_epochs": 2})
    assert plan.train_config["save_every_n_epochs"] == 2
    assert "only the final LoRA" not in capsys.readouterr().out
    # Step saves keep a state to resume from, so no note.
    _plan(image_dataset, tmp_path, form_values={"save_every_n_epochs": 0}, config_overrides={"save_every_n_steps": 200})
    assert "only the final LoRA" not in capsys.readouterr().out


@needs_submodule
def test_vpred_base_pick_reports_dropped_form_values(image_dataset, tmp_path, capsys):
    """The v-pred flags replace min_snr and noise_offset, so the plan tells the user."""
    train.build_plan(
        ROUTE,
        ROUTE.base("noobai-xl-vpred-1.0"),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws",
        project_name="test-lora",
        config_overrides={"min_snr_gamma": 5, "noise_offset": 0.03},
        sd_scripts_dir=SD_SCRIPTS,
    )
    out = capsys.readouterr().out
    assert "min_snr_gamma dropped" in out and "noise_offset dropped" in out
    _plan(image_dataset, tmp_path, config_overrides={"min_snr_gamma": 5})  # eps base: nothing to report
    assert "dropped" not in capsys.readouterr().out


@needs_submodule
def test_wandb_opt_in_wires_native_logging(image_dataset, tmp_path, monkeypatch):
    """wandb needs no logging_dir and reads WANDB_API_KEY from the environment.
    The W&B project takes the LoRA project name."""
    monkeypatch.setenv("WANDB_API_KEY", "k")
    plan = _plan(image_dataset, tmp_path, log_to_wandb=True)
    assert plan.train_config["log_with"] == "wandb"
    assert plan.train_config["log_tracker_name"] == "test-lora"
    assert "wandb_api_key" not in plan.train_config
    assert "logging_dir" not in plan.train_config
    plan = _plan(image_dataset, tmp_path)
    assert "log_with" not in plan.train_config


@needs_submodule
def test_wandb_without_credentials_refused_at_plan_time(image_dataset, tmp_path, monkeypatch):
    """A missing wandb login stops the run only after caching, so the plan refuses it."""
    monkeypatch.setattr(train.environment, "wandb_login_available", lambda: False)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        _plan(image_dataset, tmp_path, log_to_wandb=True)
    # The same check when wandb is asked for through config_overrides.
    for log_with in ("wandb", "all"):
        with pytest.raises(ValueError, match=f"log_with is '{log_with}'.*remove log_with"):
            _plan(image_dataset, tmp_path, config_overrides={"log_with": log_with})
    plan = _plan(image_dataset, tmp_path, config_overrides={"log_with": "tensorboard", "logging_dir": "logs"})
    assert plan.train_config["log_with"] == "tensorboard"


def test_execute_installs_wandb_on_demand(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train.environment, "ensure_wandb", lambda: calls.append(1))
    plan = _fake_plan([sys.executable, "-c", "print('ok')"], tmp_path)
    train.execute(plan, sd_scripts_dir=tmp_path)
    assert calls == []  # no wandb in the plan, no install
    plan.train_config["log_with"] = "wandb"
    train.execute(plan, sd_scripts_dir=tmp_path)
    assert calls == [1]


@needs_submodule
def test_bf16_route_on_t4_gets_the_fit_hint(image_dataset, tmp_path, capsys, monkeypatch):
    """Anima's preset (bf16, 1024, batch 2) runs out of memory on a T4 at the first step.
    The hint prints at plan time, only for a plan too big for a T4."""
    hint = "a T4 does not support natively"
    monkeypatch.setattr(train.environment, "gpu_name", lambda: "Tesla T4")
    _anima_plan(image_dataset, tmp_path)
    out = capsys.readouterr().out
    assert hint in out
    assert "quota" not in out and "free" not in out
    _anima_plan(image_dataset, tmp_path, resolution=768, train_batch_size=1)
    assert hint not in capsys.readouterr().out  # already sized for a T4
    _plan(image_dataset, tmp_path)  # an fp16 preset
    assert hint not in capsys.readouterr().out
    _anima_plan(image_dataset, tmp_path, config_overrides={"mixed_precision": "fp16"})
    assert hint not in capsys.readouterr().out  # keyed on the precision the run uses
    for detected in (lambda: "NVIDIA L4", lambda: None):  # an L4, or no GPU at all
        monkeypatch.setattr(train.environment, "gpu_name", detected)
        _anima_plan(image_dataset, tmp_path)
        assert hint not in capsys.readouterr().out


@needs_submodule
def test_custom_base_is_vpred_gives_a_pasted_base_the_vpred_flags(image_dataset, tmp_path, capsys):
    pasted = models.custom_base("someone/noob-vpred-merge:merge.safetensors")
    common = dict(
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws",
        project_name="test-lora",
        form_values={"min_snr_gamma": 5},
        sd_scripts_dir=SD_SCRIPTS,
    )
    plan = train.build_plan(ROUTE, pasted, **common)
    assert "v_parameterization" not in plan.train_config
    plan = train.build_plan(ROUTE, pasted, custom_base_is_vpred=True, **common)
    for key, value in config.VPRED_FLAGS.items():
        assert plan.train_config[key] == value, key
    assert "min_snr_gamma" not in plan.train_config
    assert "v-prediction" in capsys.readouterr().out
    # A model from the list keeps its own setting.
    plan = train.build_plan(ROUTE, ROUTE.base(), custom_base_is_vpred=True, **common)
    assert "v_parameterization" not in plan.train_config
    assert "does nothing for a model from the list" in capsys.readouterr().out


@needs_submodule
def test_quick_train_carries_the_oneclick_knobs(image_dataset, tmp_path, fake_download):
    """The 1-click training fields reach the plan, and the optimizer's recommended settings apply."""
    plan = train.quick_train(
        model="illustrious",
        dataset=image_dataset,
        project_name="knobs",
        workspace=tmp_path / "runs",
        auto_tag=False,
        dry_run=True,
        optimizer="Prodigy",
        form_values={
            "learning_rate": 3e-4,
            "max_train_epochs": 10,
            "max_train_steps": 0,
            "lr_scheduler": "constant_with_warmup",
            "lr_warmup_steps": 0.05,
            "lr_scheduler_num_cycles": 3,
            "min_snr_gamma": 0,
            "save_every_n_epochs": 2,
        },
        config_overrides={},
        dataset_overrides={},
    )
    config = plan.train_config
    assert config["optimizer_type"] == "Prodigy" and config["learning_rate"] == 1.0
    assert config["lr_scheduler"] == "constant_with_warmup" and config["lr_warmup_steps"] == 0.05
    assert config["save_every_n_epochs"] == 2
    assert "min_snr_gamma" not in config and "max_train_steps" not in config


@needs_submodule
@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(optimizer="Prodigyy"), "Did you mean Prodigy"),
        (dict(config_overrides={"optimizer_type": "AdamW9bit"}), "Did you mean AdamW8bit"),
        (dict(form_values={"lr_scheduler": "cosin"}), "Did you mean cosine"),
        (dict(form_values={"lr_scheduler": "cosine_with_min_lr"}), "lr_scheduler_min_lr_ratio"),
        (dict(config_overrides={"lr_scheduler": "greedy"}), "cannot be used here"),
        (dict(sample_prompt="/no/such/prompts.txt"), "does not exist"),
        (dict(log_to_wandb=True), "log_to_wandb is on"),
        (dict(config_overrides={"log_with": "all"}), "log_with is 'all'"),
        (dict(dataset_overrides={"learning_rate": 1}), "dataset_overrides cannot take"),
        (dict(config_overrides={"train_data_dir": "x"}), "the trainer ignores"),
    ],
)
def test_quick_train_refuses_typos_before_any_download(image_dataset, tmp_path, monkeypatch, kwargs, message):
    def no_download(*args, **kw):
        raise AssertionError("downloaded before refusing")

    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(SD_SCRIPTS))
    monkeypatch.setattr(train.environment, "wandb_login_available", lambda: False)
    monkeypatch.setattr(train.models, "prepare_base", no_download)
    monkeypatch.setattr(train.dataset_mod, "prepare_dataset", no_download)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        train.quick_train(model="sdxl", dataset=image_dataset, project_name="p", workspace=tmp_path / "runs", **kwargs)


@needs_submodule
def test_the_early_checks_skip_what_the_run_does_not_use(image_dataset, tmp_path, fake_download):
    """A schedule-free optimizer and lr_scheduler_type both leave lr_scheduler unused."""
    for kwargs in (
        dict(optimizer="ProdigyScheduleFree", form_values={"lr_scheduler": "cosin"}),
        dict(form_values={"lr_scheduler": "cosin"}, config_overrides={"lr_scheduler_type": "CosineAnnealingLR"}),
    ):
        train.quick_train(
            model="sdxl", dataset=image_dataset, project_name="p", workspace=tmp_path / "runs", dry_run=True, **kwargs
        )


# Where each swapping trainer's block count and limit live at the pin.
_BLOCK_COUNTS = {
    "anima": ("library/anima_utils.py", r'"num_blocks": (\d+)', "library/anima_models.py", "self.num_blocks - 2"),
    "moss_image": (
        "library/moss_image_models.py",
        r'"num_layers": (\d+)',
        "library/moss_image_models.py",
        "self.num_layers - 2",
    ),
}


@needs_submodule
def test_block_swap_limits_and_conflicts_match_the_trainers():
    swapping = {key for key, route in registry.ROUTES.items() if route.max_blocks_to_swap}
    assert swapping == set(_BLOCK_COUNTS)
    for key, (config_file, count_pattern, model_file, limit) in _BLOCK_COUNTS.items():
        route = registry.get(key)
        count = int(re.search(count_pattern, (SD_SCRIPTS / config_file).read_text(encoding="utf-8")).group(1))
        assert f"self.blocks_to_swap <= {limit}" in (SD_SCRIPTS / model_file).read_text(encoding="utf-8")
        assert route.max_blocks_to_swap == count - 2, key
        trainer = (SD_SCRIPTS / route.entrypoint).read_text(encoding="utf-8")
        for conflict in train.BLOCK_SWAP_CONFLICTS:
            assert re.search(rf"blocks_to_swap[^\n]*\n?[^\n]*not args\.{conflict}|args\.{conflict}:", trainer), conflict
        # The trainer freezes the base before it builds the one-way offloader.
        assert "requires_grad_(False)\n            swap_config = BlockSwapConfig.from_args" in trainer
    offloading = (SD_SCRIPTS / "library" / "custom_offloading_utils.py").read_text(encoding="utf-8")
    assert 'if h2d_only and supports_backward and not getattr(args, "gradient_checkpointing", False)' in offloading


@needs_submodule
def test_blocks_to_swap_turns_on_one_way_copies_and_refuses_what_the_trainer_stops_on(image_dataset, tmp_path, capsys):
    plan = _anima_plan(image_dataset, tmp_path, form_values={"blocks_to_swap": 0})
    assert "blocks_to_swap" not in plan.train_config and "block_swap_h2d_only" not in plan.train_config
    plan = _anima_plan(image_dataset, tmp_path, form_values={"blocks_to_swap": 8})
    assert plan.train_config["blocks_to_swap"] == 8 and plan.train_config["block_swap_h2d_only"] is True
    assert "so block_swap_h2d_only is on too" in capsys.readouterr().out
    # The user's own value wins, and without gradient checkpointing there is no default.
    plan = _anima_plan(
        image_dataset, tmp_path, form_values={"blocks_to_swap": 8}, config_overrides={"block_swap_h2d_only": False}
    )
    assert plan.train_config["block_swap_h2d_only"] is False
    plan = _anima_plan(
        image_dataset, tmp_path, form_values={"blocks_to_swap": 8}, config_overrides={"gradient_checkpointing": None}
    )
    assert "block_swap_h2d_only" not in plan.train_config
    refused = [
        ({"blocks_to_swap": 27}, "at most 26 blocks"),
        ({"blocks_to_swap": 8, "cpu_offload_checkpointing": True}, "cannot be used with cpu_offload_checkpointing"),
        ({"blocks_to_swap": 8, "unsloth_offload_checkpointing": True}, "with unsloth_offload_checkpointing"),
        ({"blocks_to_swap": 8, "block_swap_h2d_only": True, "gradient_checkpointing": None}, "needs gradient_che"),
    ]
    for overrides, message in refused:
        with pytest.raises(ValueError, match=message):
            _anima_plan(image_dataset, tmp_path, config_overrides=overrides)
        # Before any download, too.
        with pytest.raises(ValueError, match=message):
            train.check_before_downloads(registry.get("anima"), {"config_overrides": overrides})
    # A route whose trainer does not swap blocks is left alone.
    assert train.check_block_swap(ROUTE, {"blocks_to_swap": 99, "gradient_checkpointing": True}, {}) == {}


def test_quick_train_checks_the_base_before_the_dataset_work(tmp_path, monkeypatch):
    """A typo in the base fails in seconds, before a .zip is unpacked or tagged."""
    touched = []
    monkeypatch.setattr(train.dataset_mod, "prepare_dataset", lambda *args, **kwargs: touched.append("dataset"))
    monkeypatch.setattr(train.tagging, "tag_images", lambda *args, **kwargs: touched.append("tagger"))
    monkeypatch.setattr(plan_mod, "check_before_downloads", lambda *args, **kwargs: None)
    common = dict(model="illustrious", dataset=tmp_path / "set.zip", project_name="p", workspace=tmp_path, dry_run=True)
    with pytest.raises(ValueError, match="model page"):
        train.quick_train(base="https://example.com/some-model-page", **common)
    with pytest.raises(KeyError, match="unknown base model"):
        train.quick_train(base="no-such-base", **common)
    with pytest.raises(ValueError, match="convrot_int8"):
        train.quick_train(model="moss_image", dataset=tmp_path / "set.zip", project_name="p", convrot_int8="maybe")
    assert touched == []


def test_quick_train_looks_up_a_civitai_link_once(image_dataset, tmp_path, monkeypatch):
    """prepare_base gets the early check's result, so its warnings print once."""
    lookups = []
    version = {
        "id": 222,
        "name": "v2.0",
        "baseModel": "Illustrious",
        "description": "",
        "files": [
            {
                "name": "great_v2.safetensors",
                "type": "Model",
                "primary": True,
                "metadata": {"format": "SafeTensor"},
                "downloadUrl": "https://civitai.com/api/download/models/222",
            }
        ],
    }

    def fake_json(url):
        lookups.append(url)
        return {"id": 99, "name": "Great", "type": "Checkpoint", "modelVersions": [version]}

    received = []

    def fake_prepare_base(route, base_key=None, custom="", **kwargs):
        received.append(custom)
        raise RuntimeError("stop before the download")

    monkeypatch.setattr(train.models, "_civitai_json", fake_json)
    monkeypatch.setattr(train.models, "prepare_base", fake_prepare_base)
    monkeypatch.setattr(plan_mod, "check_before_downloads", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.environment, "scratch_dir", lambda name: tmp_path / "scratch" / name)
    with pytest.raises(RuntimeError, match="stop before the download"):
        train.quick_train(
            model="illustrious",
            base="https://civitai.com/models/99/great",
            dataset=image_dataset,
            project_name="p",
            workspace=tmp_path / "runs",
            trigger_word="z",
            dry_run=True,
        )
    assert len(lookups) == 1
    assert isinstance(received[0], models.BaseModel) and received[0].filename == "great_v2.safetensors"

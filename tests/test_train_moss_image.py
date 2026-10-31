"""build_plan and quick_train on MOSS-Image, and the ConvRot INT8 choice."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from kohya_trainer import models, registry, train

from conftest import SD_SCRIPTS, needs_submodule


def test_moss_image_preview_is_a_sentence():
    # MOSS-Image learned from sentences, so its preview is one, with the quality tags at the end.
    route = registry.get("moss_image")
    scene = "looking at the viewer. A cowboy shot against a simple background, best quality."
    assert train.default_sample_prompt(route, "zkx") == f"zkx, alone, {scene}"
    assert train.default_sample_prompt(route) == f"A character, alone, {scene}"
    tagged = train.default_sample_prompt(route, "zkx", ["1girl", "zkx", "blue dress"])
    assert tagged == f"zkx, 1girl, blue dress, {scene}"
    assert train.default_sample_prompt(route, "", ["1girl", "blue dress"]) == f"1girl, blue dress, {scene}"
    # The tag routes keep their tag lists.
    assert not any(registry.get(key).sample_sentence for key in registry.ROUTES if key != "moss_image")


# --- the MOSS-Image route ---


MOSS = registry.get("moss_image")


def _moss_plan(image_dataset, tmp_path, **kwargs) -> train.LaunchPlan:
    text_encoder = tmp_path / "qwen3.5_9b_bf16.safetensors"
    text_encoder.touch()
    defaults = dict(
        checkpoint_path=Path("/models/moss_image_base_bf16.safetensors"),
        vae_path=Path("/models/flux2-vae.safetensors"),
        text_encoder_path=text_encoder,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws-moss",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    defaults.update(kwargs)
    return train.build_plan(MOSS, MOSS.base(), **defaults)


@needs_submodule
def test_moss_image_plan_uses_its_recipe(image_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    plan = _moss_plan(image_dataset, tmp_path)
    config_ = plan.train_config
    assert plan.command[3].endswith("moss_image_train_network.py")
    assert config_["network_module"] == "networks.lora_generic"
    assert (config_["network_dim"], config_["network_alpha"], config_["learning_rate"]) == (32, 32, 1e-4)
    assert config_["max_train_epochs"] == 7
    assert (config_["timestep_sampling"], config_["discrete_flow_shift"]) == ("shift", 3.0)
    assert config_["text_encoder"] == str(tmp_path / "qwen3.5_9b_bf16.safetensors")
    assert config_["vae"].endswith("flux2-vae.safetensors")
    assert config_["network_train_unet_only"] is True and config_["cache_text_encoder_outputs_to_disk"] is True
    assert not any(key.startswith("fp8") for key in config_)
    # The dataset defaults the dataset config has no place for are training settings.
    assert config_["enable_multiline_caption"] is True and config_["caption_dropout_rate"] == 0.1
    general = plan.dataset_config["general"]
    assert "enable_multiline_caption" not in general and "caption_dropout_rate" not in general
    assert plan.dataset_config["datasets"][0]["batch_size"] == 1
    # The user's own values win, from either dict.
    plan = _moss_plan(image_dataset, tmp_path, dataset_overrides={"caption_dropout_rate": 0})
    assert plan.train_config["caption_dropout_rate"] == 0
    plan = _moss_plan(image_dataset, tmp_path, config_overrides={"enable_multiline_caption": None}, train_batch_size=2)
    assert "enable_multiline_caption" not in plan.train_config
    assert plan.dataset_config["datasets"][0]["batch_size"] == 2


@needs_submodule
def test_moss_image_aims_for_200_images_per_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    data = tmp_path / "many"
    data.mkdir()
    for i in range(25):
        (data / f"{i}.png").write_bytes(b"\x89PNG fake")
        (data / f"{i}.txt").write_text("1girl", encoding="utf-8")
    plan = _moss_plan(data, tmp_path)
    assert plan.dataset_config["datasets"][0]["subsets"][0]["num_repeats"] == 8  # 200 / 25


@needs_submodule
def test_moss_image_keeps_caption_candidates_with_text_caching(image_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    (image_dataset / "a.txt").write_text("1girl, solo\n---===--- # tags\nA girl stands alone.", encoding="utf-8")
    plan = _moss_plan(image_dataset, tmp_path)
    assert plan.train_config["cache_text_encoder_outputs"] is True
    # Only a # comment can follow the separator. The trainer stops on other text while loading.
    (image_dataset / "a.txt").write_text("1girl, solo\n---===--- short\nA girl stands alone.", encoding="utf-8")
    with pytest.raises(ValueError, match=r"caption for a\.png has a separator line with text after it"):
        _moss_plan(image_dataset, tmp_path)
    with pytest.raises(ValueError, match="only a # comment can follow the separator"):
        train.check_before_downloads(MOSS, {}, dataset_dir=image_dataset)
    # Without enable_multiline_caption the backend reads only the first line.
    _moss_plan(image_dataset, tmp_path, config_overrides={"enable_multiline_caption": None})


@needs_submodule
def test_moss_image_samples_are_a_sentence_and_keep_an_empty_negative(image_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    line = _moss_plan(image_dataset, tmp_path, trigger_word="zkx").sample_prompts_file.read_text(encoding="utf-8")
    assert line.startswith(
        "zkx, 1girl, solo, smile, looking at the viewer. A cowboy shot against a simple background, best quality. --n "
    )
    assert " --n nsfw, lowres, worst quality, bad anatomy, watermark, jpeg artifact" in line
    assert " --s 30" in line and " --l 4 " in line and " --w 832 --h 1216" in line
    line = _moss_plan(image_dataset, tmp_path, sample_negative_prompt="").sample_prompts_file.read_text(
        encoding="utf-8"
    )
    assert " --n" not in line


@needs_submodule
def test_moss_image_preview_never_takes_the_caption_separator_for_a_tag(image_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    for name in ("a", "b"):
        (image_dataset / f"{name}.txt").write_text("1girl, solo\n@@@\n1girl, blue dress", encoding="utf-8")
    plan = _moss_plan(image_dataset, tmp_path, config_overrides={"multiline_caption_separator": "@@@"})
    line = plan.sample_prompts_file.read_text(encoding="utf-8")
    assert line.startswith("1girl, solo, blue dress, looking at the viewer.")
    assert "@@@" not in line


@needs_submodule
def test_the_prompt_flags_we_write_are_ones_the_moss_image_sampler_reads():
    """library/sampling.py turns each flag into a key, and library/moss_image_train_utils.py reads the keys."""
    parser = (SD_SCRIPTS / "library" / "sampling.py").read_text(encoding="utf-8")
    sampler = (SD_SCRIPTS / "library" / "moss_image_train_utils.py").read_text(encoding="utf-8")
    keys = {"n": "negative_prompt", "w": "width", "h": "height", "s": "sample_steps", "l": "scale", "d": "seed"}
    for flag, key in keys.items():
        assert re.search(rf're\.match\(r"{flag} [^\n]*\n[^\n]*\n\s*prompt_dict\["{key}"\]', parser), flag
        assert f'prompt_dict.get("{key}"' in sampler, key


@needs_submodule
def test_moss_image_warns_on_a_t4(image_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: "Tesla T4")
    monkeypatch.setattr(models.environment, "gpu_memory_gb", lambda: 15.0)
    order = []

    def fake_download(spec):
        order.append(("download", "warning: MOSS-Image is not practical on a T4" in capsys.readouterr().out))
        return tmp_path / "file.safetensors"

    monkeypatch.setattr(models, "download", fake_download)
    models.prepare_base(MOSS)
    # The warning comes before the first download, which is large here.
    assert order[0] == ("download", True)
    _moss_plan(image_dataset, tmp_path)
    out = capsys.readouterr().out
    assert "not practical on a T4" not in out  # once is enough
    assert "use train_batch_size = 1" not in out  # the bf16 batch advice is for routes that fit a T4


@needs_submodule
def test_moss_image_text_encoder_lora_with_text_caching_is_refused_before_downloads(
    image_dataset, tmp_path, monkeypatch
):
    """The trainer asserts this pair after it loads the dataset."""
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    te_lora = {"network_train_unet_only": False}
    with pytest.raises(ValueError, match="requires network_train_unet_only"):
        train.check_before_downloads(MOSS, {"config_overrides": te_lora})
    with pytest.raises(ValueError, match="requires network_train_unet_only"):
        _moss_plan(image_dataset, tmp_path, config_overrides=te_lora)
    no_cache = {"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None}
    train.check_before_downloads(MOSS, {"config_overrides": {**te_lora, **no_cache}})
    capsys_free = _moss_plan(image_dataset, tmp_path, config_overrides={**te_lora, **no_cache})
    assert capsys_free.train_config["network_train_unet_only"] is False
    # A text-encoder rate of 0 trains the DiT only, so caching stays on.
    train.check_before_downloads(MOSS, {"config_overrides": {**te_lora, "text_encoder_lr": 0}})


@needs_submodule
def test_turning_caching_off_for_a_text_encoder_lora_gets_no_memory_note(image_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    no_cache = {"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None}
    _moss_plan(image_dataset, tmp_path, config_overrides=no_cache)
    assert "Remove the override to save GPU memory" in capsys.readouterr().out
    _moss_plan(image_dataset, tmp_path, config_overrides={**no_cache, "network_train_unet_only": False})
    assert "Remove the override" not in capsys.readouterr().out


@needs_submodule
def test_moss_image_text_embedder_switch_and_its_rate(image_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    plan = _moss_plan(image_dataset, tmp_path, config_overrides={"network_args": ["train_text_embedder=True"]})
    assert "network_reg_lrs=.*text_embedder.*=5e-5" in plan.train_config["network_args"]
    assert "the text embedder's learning rate is set to 5e-5" in capsys.readouterr().out
    # Upstream's lora_generic takes the Anima name too.
    plan = _moss_plan(image_dataset, tmp_path, config_overrides={"network_args": ["train_llm_adapter=True"]})
    assert "network_reg_lrs=.*text_embedder.*=5e-5" in plan.train_config["network_args"]
    # LoHa reads only the Anima name.
    with pytest.raises(ValueError, match=r'networks.loha on MOSS-Image.*"train_llm_adapter=True"'):
        _moss_plan(
            image_dataset,
            tmp_path,
            network_type="loha",
            config_overrides={"network_args": ["train_text_embedder=True"]},
        )
    with pytest.raises(ValueError, match="at most 40 blocks"):
        _moss_plan(image_dataset, tmp_path, form_values={"blocks_to_swap": 41})


@needs_submodule
def test_convrot_flags_ride_the_base_into_the_config(image_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    pasted = dataclasses.replace(
        models.custom_base("someone/moss-merge:merge.safetensors"), convrot_flags=("convrot_int8",)
    )
    text_encoder = tmp_path / "qwen3.5_9b_int8_convrot.safetensors"
    text_encoder.touch()
    plan = train.build_plan(
        MOSS,
        pasted,
        checkpoint_path=Path("/models/merge.safetensors"),
        vae_path=None,
        text_encoder_path=text_encoder,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws-int8",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    assert plan.train_config["convrot_int8"] is True
    assert "convrot_int8_text_encoder" not in plan.train_config  # the int8 file is detected by the backend
    # Both flags ride the base when both models are converted while loading.
    both = dataclasses.replace(pasted, convrot_flags=("convrot_int8", "convrot_int8_text_encoder"))
    plan = train.build_plan(
        MOSS,
        both,
        checkpoint_path=Path("/models/merge.safetensors"),
        vae_path=None,
        text_encoder_path=text_encoder,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws-int8-both",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    assert plan.train_config["convrot_int8"] is True
    assert plan.train_config["convrot_int8_text_encoder"] is True
    # Without flags on the base, no flag reaches the config.
    assert "convrot_int8" not in _moss_plan(image_dataset, tmp_path).train_config
    # bf16 asked for after the int8 file downloaded gets a note: the backend uses the int8 file.
    capsys.readouterr()
    _moss_plan(
        image_dataset,
        tmp_path,
        checkpoint_path=Path("/models/moss_image_base_int8_convrot.safetensors"),
        config_overrides={"convrot_int8": False},
    )
    assert "moss_image_base_int8_convrot.safetensors is already a ConvRot INT8 file" in capsys.readouterr().out


@needs_submodule
def test_quick_train_hands_the_convrot_field_and_overrides_to_the_download(image_dataset, tmp_path, monkeypatch):
    seen = {}
    fake = tmp_path / "fake.safetensors"
    fake.write_bytes(b"00")

    def prepare_base(route, base_key=None, custom="", convrot_int8="auto", config_overrides=None):
        seen.update(convrot_int8=convrot_int8, config_overrides=config_overrides)
        return route.base(base_key), fake, fake, fake

    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(SD_SCRIPTS))
    monkeypatch.setattr(train.environment, "gpu_name", lambda: None)
    monkeypatch.setattr(train.models, "prepare_base", prepare_base)
    train.quick_train(
        model="moss_image",
        dataset=image_dataset,
        project_name="p",
        workspace=tmp_path / "runs",
        auto_tag=False,
        dry_run=True,
        convrot_int8="on",
        config_overrides={"convrot_int8_text_encoder": False},
    )
    assert seen == {"convrot_int8": "on", "config_overrides": {"convrot_int8_text_encoder": False}}


def test_moss_image_block_22_note_pastes_into_a_working_pattern():
    # The note's config_overrides entry gives the backend a pattern that drops block 22 only.
    import ast

    note = registry.get("moss_image").tuning_note
    entry = re.search(r'`("network_args": \["exclude_patterns=\[.layers.*?\]"\])`', note).group(1)
    overrides = ast.literal_eval("{" + entry + "}")
    key, value = overrides["network_args"][0].split("=", 1)  # as train_network.py splits it
    assert key == "exclude_patterns"
    (pattern,) = ast.literal_eval(value)  # as networks/lora_generic.py reads it
    assert re.fullmatch(pattern, "layers.22.attn.to_q")
    for name in ("layers.2.attn.to_q", "layers.220.attn.to_q", "layers.12.ffn.w1", "text_embedder.proj"):
        assert not re.fullmatch(pattern, name), name
    # Two patterns share one entry: the backend keeps only the last entry of a name.
    both = re.search(r'`("network_args": \["exclude_patterns=\[[^"]*, [^"]*\]"\])`', note).group(1)
    entries = ast.literal_eval("{" + both + "}")["network_args"]
    kwargs = dict(entry.split("=", 1) for entry in entries)
    assert ast.literal_eval(kwargs["exclude_patterns"]) == [pattern, ".*ffn.*"]

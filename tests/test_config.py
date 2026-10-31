from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from kohya_trainer import config, registry

from conftest import SD_SCRIPTS, needs_submodule

ROUTE = registry.get("illustrious")
EPS_BASE = ROUTE.base("illustrious-xl-v0.1")
VPRED_BASE = ROUTE.base("noobai-xl-vpred-1.0")


def _build(base=EPS_BASE, **kwargs):
    return config.build_train_config(
        ROUTE,
        base,
        checkpoint_path="/models/base.safetensors",
        output_dir="/out",
        output_name="test-lora",
        **kwargs,
    )


def test_preset_loads_flat():
    preset = config.load_preset("sdxl_lora.toml")
    assert preset["network_module"] == "networks.lora"
    assert preset["mixed_precision"] == "fp16"
    # flattened: no nested tables survive
    assert not any(isinstance(v, dict) for v in preset.values())


def test_flatten_rejects_duplicate_leaves():
    with pytest.raises(ValueError, match="duplicate key"):
        config.flatten({"a": {"x": 1}, "b": {"x": 2}})


def test_run_values_land():
    cfg = _build()
    assert cfg["pretrained_model_name_or_path"] == "/models/base.safetensors"
    assert cfg["output_name"] == "test-lora"
    assert "v_parameterization" not in cfg  # eps base gets no vpred flags


def test_vpred_base_injects_flag_bundle():
    cfg = _build(base=VPRED_BASE)
    assert cfg["v_parameterization"] is True
    assert cfg["zero_terminal_snr"] is True
    assert cfg["scale_v_pred_loss_like_noise_pred"] is True
    # v-pred runs use debiased estimation, which does min_snr_gamma's job, so that is dropped.
    assert cfg["debiased_estimation_loss"] is True
    assert "min_snr_gamma" not in cfg
    # euler_a previews come out wrong on v-pred (the NoobAI card names Euler).
    assert cfg["sample_sampler"] == "euler"
    eps = _build()
    assert eps["sample_sampler"] == "euler_a"  # preset value untouched


def test_eps_base_keeps_min_snr_without_debias():
    cfg = _build()
    assert cfg["min_snr_gamma"] == 5
    assert "debiased_estimation_loss" not in cfg


def test_snr_debiasers_never_stack():
    """A v-pred base must not use the form's min_snr with the debiased estimation its flags add."""
    cfg = _build(base=VPRED_BASE, overrides={"min_snr_gamma": 8})
    assert "min_snr_gamma" not in cfg
    assert cfg["debiased_estimation_loss"] is True
    # To use min-SNR on a v-pred base, remove debiased_estimation_loss.
    cfg = _build(base=VPRED_BASE, overrides={"min_snr_gamma": 8, "debiased_estimation_loss": None})
    assert cfg["min_snr_gamma"] == 8


def test_noise_offset_dropped_on_ztsnr_base():
    """Offset noise imitates what a zero-terminal-SNR schedule does properly,
    so the form's value must not reach a v-pred base."""
    cfg = _build(base=VPRED_BASE, overrides={"noise_offset": 0.03})
    assert "noise_offset" not in cfg
    cfg = _build(overrides={"noise_offset": 0.03})
    assert cfg["noise_offset"] == 0.03


def test_overrides_win_and_none_deletes():
    cfg = _build(overrides={"learning_rate": 5e-5, "min_snr_gamma": None})
    assert cfg["learning_rate"] == 5e-5
    assert "min_snr_gamma" not in cfg


@needs_submodule
def test_built_config_passes_real_arg_surface():
    # Every route's preset must validate against the pinned backend surface.
    for route in registry.ROUTES.values():
        config.build_train_config(
            route,
            route.base(),
            checkpoint_path="/models/base.safetensors",
            output_dir="/out",
            output_name="test-lora",
            sd_scripts_dir=SD_SCRIPTS,
        )


@needs_submodule
def test_typo_key_is_caught():
    with pytest.raises(ValueError, match="networ_dim"):
        _build(overrides={"networ_dim": 32}, sd_scripts_dir=SD_SCRIPTS)


@needs_submodule
@pytest.mark.parametrize("key", ["convrot_int8", "text_encoder", "moss_max_token_length"])
def test_moss_image_keys_are_refused_on_other_routes(key):
    # Only moss_image_train_network.py registers these. Other trainers ignore them without a word.
    others = {route.entrypoint for route in registry.ROUTES.values()} - {"moss_image_train_network.py"}
    assert {"anima_train_network.py", "sdxl_train_network.py"} <= others
    for entrypoint in sorted(others):
        with pytest.raises(ValueError, match=key):
            config.validate_keys({key: True}, SD_SCRIPTS, entrypoint)
    config.validate_keys({key: True}, SD_SCRIPTS, "moss_image_train_network.py")


def test_compat_guard_blocks_shuffle_with_te_cache():
    train_cfg = {"cache_text_encoder_outputs": True}
    dataset_cfg = {"general": {"shuffle_caption": True}, "datasets": [{"subsets": [{}]}]}
    with pytest.raises(ValueError, match="shuffle_caption") as error:
        config.check_compat(train_cfg, dataset_cfg)
    # The message says how to turn caching off.
    assert "'cache_text_encoder_outputs': None" in str(error.value)
    config.check_compat({"cache_text_encoder_outputs": False}, dataset_cfg)  # fine when caching is off


def test_compat_guard_blocks_token_warmup_with_te_cache():
    # The trainers refuse it with cached text (is_text_encoder_output_cacheable).
    train_cfg = {"cache_text_encoder_outputs": True, "network_train_unet_only": True, "token_warmup_step": 10}
    with pytest.raises(ValueError, match="token_warmup_step"):
        config.check_compat(train_cfg, {"general": {}, "datasets": [{"subsets": [{}]}]}, "anima_train_network.py")


def test_caption_dropout_with_te_cache_only_where_the_trainer_keeps_it():
    """Anima's cache stores each image's dropout rate. SDXL's trainer asserts."""
    train_cfg = {"cache_text_encoder_outputs": True, "network_train_unet_only": True}
    dataset_cfg = {"general": {}, "datasets": [{"subsets": [{"caption_dropout_rate": 0.1}]}]}
    config.check_compat(train_cfg, dataset_cfg, "anima_train_network.py")
    with pytest.raises(ValueError, match="caption_dropout_rate"):
        config.check_compat(train_cfg, dataset_cfg, "sdxl_train_network.py")
    # Shuffle and tag dropout stay refused on Anima.
    for key, value in (("shuffle_caption", True), ("caption_tag_dropout_rate", 0.1)):
        with pytest.raises(ValueError, match=key):
            config.check_compat(
                train_cfg, {"general": {key: value}, "datasets": [{"subsets": [{}]}]}, "anima_train_network.py"
            )


def test_compat_guard_reads_the_training_config_last():
    """The backend falls back to the training config's value when the dataset
    config does not set one, and a dataset value wins over it."""
    quiet_dataset = {"general": {}, "datasets": [{"subsets": [{}]}]}
    train_cfg = {"cache_text_encoder_outputs": True, "network_train_unet_only": True, "caption_dropout_rate": 0.1}
    with pytest.raises(ValueError, match="caption_dropout_rate"):
        config.check_compat(train_cfg, quiet_dataset, "sdxl_train_network.py")
    shuffle_off = {"general": {"shuffle_caption": False}, "datasets": [{"subsets": [{}]}]}
    config.check_compat(
        {"cache_text_encoder_outputs": True, "network_train_unet_only": True, "shuffle_caption": True}, shuffle_off
    )


def test_compat_guard_blocks_te_cache_without_unet_only():
    """The trainers check this pair only after the dataset loads. When both apply, shuffle is reported first."""
    quiet_dataset = {"general": {}, "datasets": [{"subsets": [{}]}]}
    with pytest.raises(ValueError, match="network_train_unet_only"):
        config.check_compat({"cache_text_encoder_outputs": True}, quiet_dataset)
    config.check_compat(
        {"cache_text_encoder_outputs": True, "network_train_unet_only": True}, quiet_dataset
    )  # the presets' pairing passes


def test_write_toml_roundtrips(tmp_path):
    cfg = _build()
    path = config.write_toml(cfg, tmp_path / "training.toml")
    assert tomllib.loads(path.read_text(encoding="utf-8")) == cfg


def test_sample_prompts_writer(tmp_path):
    path = config.write_sample_prompts("1girl --w 1024", tmp_path / "p.txt")
    assert path.read_text(encoding="utf-8") == "1girl --w 1024\n"

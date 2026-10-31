"""Registry rows must stay true against the pinned backend."""

from __future__ import annotations

from importlib import resources

import pytest

from kohya_trainer import registry

from conftest import SD_SCRIPTS, needs_submodule


def test_route_keys_are_unique_and_wellformed():
    for key, route in registry.ROUTES.items():
        assert key == route.key
        assert route.backend == "sd-scripts"
        assert route.gpu == "L4"


@needs_submodule
def test_entrypoints_exist_in_pinned_submodule():
    for route in registry.ROUTES.values():
        assert (SD_SCRIPTS / route.entrypoint).is_file(), f"{route.key}: missing {route.entrypoint}"


def test_presets_exist():
    for route in registry.ROUTES.values():
        assert (resources.files("kohya_trainer") / "presets" / route.preset).is_file()


def test_base_models_are_coherent():
    for route in registry.ROUTES.values():
        keys = [b.key for b in route.base_models]
        assert len(keys) == len(set(keys)), f"{route.key}: duplicate base keys"
        assert route.default_base in keys
        for base in route.base_models:
            assert base.repo_id.count("/") == 1
            if base.filename:
                assert base.filename.endswith(".safetensors")
        assert route.base(route.default_base).key == route.default_base


def test_unknown_lookups_raise():
    with pytest.raises(KeyError, match="unknown model route"):
        registry.get("nope")
    with pytest.raises(KeyError, match="unknown base model"):
        registry.get("sdxl").base("nope")


def test_text_encoder_fields_are_paired():
    for route in registry.ROUTES.values():
        assert bool(route.default_text_encoder) == bool(route.text_encoder_arg), (
            f"{route.key}: default_text_encoder and text_encoder_arg must be set together"
        )


def test_vpred_base_is_flagged():
    illustrious = registry.get("illustrious")
    assert illustrious.base("noobai-xl-vpred-1.0").v_prediction
    assert not illustrious.base("illustrious-xl-v0.1").v_prediction

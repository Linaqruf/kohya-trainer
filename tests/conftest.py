from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS = ROOT / "external" / "sd-scripts"

# Allow running the suite without an editable install.
sys.path.insert(0, str(ROOT / "src"))

from kohya_trainer import models, registry, train

needs_submodule = pytest.mark.skipif(
    not (SD_SCRIPTS / "train_network.py").is_file(),
    reason="sd-scripts submodule not checked out (git submodule update --init)",
)


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    """A tiny fake dataset: three 'images', two with captions."""
    dataset = tmp_path / "data"
    dataset.mkdir()
    for name in ("a", "b", "c"):
        (dataset / f"{name}.png").write_bytes(b"\x89PNG fake")
    (dataset / "a.txt").write_text("1girl, solo", encoding="utf-8")
    (dataset / "b.txt").write_text("1girl, smile", encoding="utf-8")
    return dataset


# Shared by the test_train*.py files.
ROUTE = registry.get("illustrious")


def _plan(image_dataset, tmp_path, **kwargs) -> train.LaunchPlan:
    defaults = dict(
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    defaults.update(kwargs)
    return train.build_plan(ROUTE, ROUTE.base(), **defaults)


def _anima_plan(image_dataset, tmp_path, **kwargs) -> train.LaunchPlan:
    anima = registry.get("anima")
    text_encoder = tmp_path / "te.safetensors"
    text_encoder.touch()
    defaults = dict(
        checkpoint_path=Path("/models/dit.safetensors"),
        vae_path=None,
        text_encoder_path=text_encoder,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws-anima",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    defaults.update(kwargs)
    return train.build_plan(anima, anima.base(), **defaults)


def _fake_plan(command, tmp_path) -> train.LaunchPlan:
    return train.LaunchPlan(
        route_key="x",
        base_key="y",
        command=command,
        config_file=tmp_path / "t.toml",
        dataset_config_file=tmp_path / "d.toml",
        sample_prompts_file=tmp_path / "p.txt",
        output_dir=tmp_path,
        estimated_steps_per_epoch=1,
        estimated_total_steps=1,
        train_config={},
        dataset_config={},
    )


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    fake_ckpt = tmp_path / "fake.safetensors"
    fake_ckpt.write_bytes(b"00")
    monkeypatch.setattr(models, "download", lambda base: fake_ckpt)
    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(SD_SCRIPTS))
    return fake_ckpt


def _sd15_plan(image_dataset, tmp_path, **kwargs) -> train.LaunchPlan:
    sd15 = registry.get("sd15")
    defaults = dict(
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws15",
        project_name="test-sd15",
        sd_scripts_dir=SD_SCRIPTS,
    )
    defaults.update(kwargs)
    return train.build_plan(sd15, sd15.base(), **defaults)

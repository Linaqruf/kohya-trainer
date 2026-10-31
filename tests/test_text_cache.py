"""Old text-encoder cache files are removed when what they encode changes. The backend does not run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from kohya_trainer import dataset, train

SDXL = "sdxl_train_network.py"
ANIMA = "anima_train_network.py"
SDXL_SUFFIX = dataset.TEXT_CACHE_SUFFIXES[SDXL]
ANIMA_SUFFIX = dataset.TEXT_CACHE_SUFFIXES[ANIMA]


def _folder(root: Path, name: str = "data") -> Path:
    folder = root / name
    folder.mkdir()
    for stem in ("a", "b", "c"):
        (folder / f"{stem}.png").write_bytes(b"\x89PNG fake")
    (folder / "a.txt").write_text("1girl, solo", encoding="utf-8")
    (folder / "b.txt").write_text("1girl, smile", encoding="utf-8")
    return folder


def _build_caches(folder: Path, suffix: str = SDXL_SUFFIX) -> None:
    """What the backend leaves next to the images after a cached run."""
    for image in folder.glob("*.png"):
        (folder / f"{image.stem}{suffix}").write_bytes(b"te")
        (folder / f"{image.stem}_1024x1024_sdxl.npz").write_bytes(b"latent")


def _caches(folder: Path, suffix: str = SDXL_SUFFIX) -> set[str]:
    return {p.name for p in folder.iterdir() if p.name.endswith(suffix) and p.is_file()}


def _checkpoint(root: Path, name: str = "base.safetensors", content: bytes = b"weights") -> Path:
    path = root / "models" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def _train_config(checkpoint: Path, **extra) -> dict:
    config = {
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "pretrained_model_name_or_path": str(checkpoint),
        "mixed_precision": "fp16",
    }
    config.update(extra)
    return config


@pytest.fixture
def setup(tmp_path):
    folder = _folder(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    config = _train_config(checkpoint)
    # A first run with no record treats old caches as stale, so record them first.
    dataset.refresh_text_caches(dataset.build_dataset_config(folder), config, SDXL)
    _build_caches(folder)
    return folder, checkpoint, config


def _refresh(folder, config, entrypoint=SDXL, **dataset_kwargs):
    dataset.refresh_text_caches(dataset.build_dataset_config(folder, **dataset_kwargs), config, entrypoint)


def test_unchanged_run_keeps_the_caches(setup, capsys):
    folder, _, config = setup
    _refresh(folder, config)
    assert _caches(folder) == {"a_te_outputs.npz", "b_te_outputs.npz", "c_te_outputs.npz"}
    assert capsys.readouterr().out == ""
    # Paths and settings that do not change the cache do not count.
    moved = _train_config(Path(config["pretrained_model_name_or_path"]), seed=7, learning_rate=1e-3, max_train_epochs=3)
    _refresh(folder, moved, num_repeats=5, batch_size=1)
    assert len(_caches(folder)) == 3


def test_caption_change_removes_the_caches(setup, capsys):
    folder, _, config = setup
    (folder / "a.txt").write_text("1girl, solo, red hair", encoding="utf-8")
    _refresh(folder, config)
    assert _caches(folder) == set()
    out = capsys.readouterr().out
    assert "removed 3 old text-encoder cache files" in out and "captions changed" in out


def test_new_caption_file_removes_the_caches(setup):
    folder, _, config = setup
    (folder / "c.txt").write_text("1boy", encoding="utf-8")
    _refresh(folder, config)
    assert _caches(folder) == set()


def test_trigger_or_prefix_change_removes_the_caches(setup):
    folder, _, config = setup
    _refresh(folder, config, caption_prefix="zkx,")
    assert _caches(folder) == set()
    _build_caches(folder)
    _refresh(folder, config, caption_prefix="zkx,")
    assert len(_caches(folder)) == 3
    _refresh(folder, config, caption_prefix="zkx, masterpiece,")
    assert _caches(folder) == set()


def test_caption_suffix_and_extension_count(setup):
    folder, _, config = setup
    # caption_suffix reaches the backend from the training config.
    _refresh(folder, dict(config, caption_suffix="best quality"))
    assert _caches(folder) == set()
    _build_caches(folder)
    (folder / "a.caption").write_text("other", encoding="utf-8")
    _refresh(folder, dict(config, caption_suffix="best quality"), caption_extension=".caption")
    assert _caches(folder) == set()


def test_class_token_fallback_counts(setup):
    folder, _, config = setup
    _refresh(folder, config, class_tokens="zkx")
    assert _caches(folder) == set()


def test_text_encoder_change_removes_the_caches(setup, tmp_path, capsys):
    folder, _, _ = setup
    other = _checkpoint(tmp_path, "other.safetensors")
    _refresh(folder, _train_config(other))
    assert _caches(folder) == set()
    assert "text encoder or its settings changed" in capsys.readouterr().out
    _build_caches(folder)
    # Same name and size, different weights.
    _checkpoint(tmp_path, "other.safetensors", b"WEIGHTS")
    _refresh(folder, _train_config(other))
    assert _caches(folder) == set()
    _build_caches(folder)
    _refresh(folder, _train_config(other, max_token_length=150))
    assert _caches(folder) == set()


def test_same_checkpoint_in_another_folder_keeps_the_caches(setup, tmp_path):
    folder, checkpoint, _ = setup
    moved = tmp_path / "elsewhere" / checkpoint.name
    moved.parent.mkdir()
    moved.write_bytes(checkpoint.read_bytes())
    _refresh(folder, _train_config(moved))
    assert len(_caches(folder)) == 3


def test_only_text_caches_are_removed(setup):
    folder, _, config = setup
    (folder / "notes.json").write_text("{}", encoding="utf-8")
    (folder / f"a{ANIMA_SUFFIX}").write_bytes(b"anima te")
    sub = folder / "sub"
    sub.mkdir()
    (sub / f"a{SDXL_SUFFIX}").write_bytes(b"te")
    (folder / f"dir{SDXL_SUFFIX}").mkdir()
    before = {p.name for p in folder.iterdir()} - _caches(folder)
    (folder / "a.txt").write_text("changed", encoding="utf-8")
    _refresh(folder, config)
    assert _caches(folder) == set()
    assert {p.name for p in folder.iterdir()} == before
    assert (sub / f"a{SDXL_SUFFIX}").is_file()
    assert (folder / "a_1024x1024_sdxl.npz").read_bytes() == b"latent"


def test_file_links_go_but_never_their_targets(setup, tmp_path):
    # The backend reads a cache through a link, so a stale link goes, but not its target.
    folder, _, config = setup
    target = tmp_path / "target.npz"
    target.write_bytes(b"keep")
    link = folder / f"link{SDXL_SUFFIX}"
    linked_dir = folder / f"dirlink{SDXL_SUFFIX}"
    try:
        os.symlink(target, link)
        os.symlink(tmp_path / "models", linked_dir, target_is_directory=True)
    except OSError:
        pytest.skip("this system cannot make symlinks")
    (folder / "a.txt").write_text("changed", encoding="utf-8")
    _refresh(folder, config)
    assert not link.is_symlink() and target.read_bytes() == b"keep"
    assert linked_dir.is_symlink(), "a link to a folder stays"
    assert (tmp_path / "models" / "base.safetensors").is_file()


@pytest.mark.parametrize(
    "entrypoint, key, value",
    [
        (SDXL, key, value)
        for key, value in (("max_token_length", 225), ("weighted_captions", True), ("mixed_precision", "fp16"))
    ]
    + [
        (ANIMA, key, value)
        for key, value in (
            ("qwen3_max_token_length", 256),
            ("t5_max_token_length", 256),
            ("t5_tokenizer_path", "/models/t5"),
            ("mixed_precision", "fp16"),
        )
    ],
)
def test_each_text_encoder_setting_counts(tmp_path, entrypoint, key, value):
    folder = _folder(tmp_path)
    suffix = dataset.TEXT_CACHE_SUFFIXES[entrypoint]
    config = {"cache_text_encoder_outputs_to_disk": True, "mixed_precision": "bf16", "qwen3": "/models/qwen3"}
    config["pretrained_model_name_or_path"] = str(_checkpoint(tmp_path))
    _refresh(folder, config, entrypoint=entrypoint)
    _build_caches(folder, suffix)
    _refresh(folder, dict(config, **{key: value}), entrypoint=entrypoint)
    assert _caches(folder, suffix) == set(), key


@pytest.mark.parametrize(
    "key, value",
    [
        ("caption_separator", ";"),
        ("secondary_separator", ";;;"),
        ("enable_wildcard", True),
        ("enable_multiline_caption", True),
        ("multiline_caption_separator", "---"),
    ],
)
def test_each_caption_setting_counts(setup, key, value):
    folder, _, config = setup
    _refresh(folder, dict(config, **{key: value}))
    assert _caches(folder) == set(), key


def test_the_backends_second_caption_name_counts(tmp_path):
    # read_caption also tries the stem without its last four "_" parts.
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a_1_2_3_4.png").write_bytes(b"\x89PNG fake")
    (folder / "a.txt").write_text("1girl", encoding="utf-8")
    config = _train_config(_checkpoint(tmp_path))
    _refresh(folder, config)
    _build_caches(folder)
    (folder / "a.txt").write_text("1boy", encoding="utf-8")
    _refresh(folder, config)
    assert _caches(folder) == set()


def test_only_the_first_caption_line_counts_unless_multiline(setup):
    folder, _, config = setup
    (folder / "a.txt").write_text("1girl, solo\nsecond line", encoding="utf-8")
    _refresh(folder, config)
    _build_caches(folder)
    (folder / "a.txt").write_text("1girl, solo\nanother second line", encoding="utf-8")
    _refresh(folder, config)
    assert len(_caches(folder)) == 3, "the backend reads only the first line"
    multiline = dict(config, enable_multiline_caption=True)
    _refresh(folder, multiline)
    _build_caches(folder)
    (folder / "a.txt").write_text("1girl, solo\nyet another line", encoding="utf-8")
    _refresh(folder, multiline)
    assert _caches(folder) == set()


def test_images_with_mixed_case_extensions_count(setup):
    # The backend's glob ignores case on Windows.
    folder, _, config = setup
    (folder / "d.Png").write_bytes(b"\x89PNG fake")
    _refresh(folder, config)
    _build_caches(folder)
    (folder / "d.txt").write_text("1boy", encoding="utf-8")
    _refresh(folder, config)
    assert _caches(folder) == set()


@pytest.mark.skipif(os.name != "nt", reason="file names ignore case only on Windows")
def test_cache_names_ignore_case_on_windows(setup):
    folder, _, config = setup
    # A stem with no cache yet, so the file system keeps this exact name.
    (folder / "d_TE_OUTPUTS.NPZ").write_bytes(b"te")
    assert "d_TE_OUTPUTS.NPZ" in os.listdir(folder)
    (folder / "a.txt").write_text("changed", encoding="utf-8")
    _refresh(folder, config)
    assert "d_TE_OUTPUTS.NPZ" not in os.listdir(folder)


def test_a_cache_that_cannot_be_removed_says_what_to_do(setup, monkeypatch):
    folder, _, config = setup
    (folder / "a.txt").write_text("changed", encoding="utf-8")

    def refuse(path):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(dataset.os, "remove", refuse)
    with pytest.raises(RuntimeError, match="could not remove the old text-encoder cache files") as error:
        _refresh(folder, config)
    assert str(folder) in str(error.value) and "cache_text_encoder_outputs_to_disk" in str(error.value)
    # The record is not updated, so the next run tries again.
    monkeypatch.undo()
    _refresh(folder, config)
    assert _caches(folder) == set()


def test_old_caches_without_a_record_are_removed(tmp_path, capsys):
    folder = _folder(tmp_path)
    _build_caches(folder)
    config = _train_config(_checkpoint(tmp_path))
    _refresh(folder, config)
    assert _caches(folder) == set()
    assert "no record" in capsys.readouterr().out
    assert (folder / dataset.TEXT_CACHE_RECORD).is_file()
    # A fresh folder with no caches: nothing removed, nothing printed.
    fresh = _folder(tmp_path, "fresh")
    _refresh(fresh, config)
    assert capsys.readouterr().out == ""


def test_record_is_never_read_as_an_image_or_caption(tmp_path):
    folder = _folder(tmp_path)
    _refresh(folder, _train_config(_checkpoint(tmp_path)))
    record = folder / dataset.TEXT_CACHE_RECORD
    assert record.name.startswith(".")
    assert record.suffix not in dataset.IMAGE_EXTENSIONS | {".txt", ".caption"}
    assert dataset.summarize(folder)["images"] == 3


def test_memory_only_cache_and_uncached_routes_do_nothing(setup):
    folder, _, config = setup
    (folder / "a.txt").write_text("changed", encoding="utf-8")
    memory_only = dict(config, cache_text_encoder_outputs_to_disk=None)
    _refresh(folder, memory_only)
    _refresh(folder, config, entrypoint="train_network.py")  # SD1.5
    assert len(_caches(folder)) == 3


def test_anima_uses_its_own_suffix_and_text_encoder(tmp_path):
    folder = _folder(tmp_path)
    qwen3 = _checkpoint(tmp_path, "qwen3.safetensors")
    config = {"cache_text_encoder_outputs_to_disk": True, "qwen3": str(qwen3), "mixed_precision": "bf16"}
    _refresh(folder, config, entrypoint=ANIMA)
    _build_caches(folder, ANIMA_SUFFIX)
    _build_caches(folder, SDXL_SUFFIX)
    # A new DiT does not touch the text cache. A new qwen3 file does.
    _refresh(folder, dict(config, pretrained_model_name_or_path="/models/new-dit.safetensors"), entrypoint=ANIMA)
    assert len(_caches(folder, ANIMA_SUFFIX)) == 3
    _refresh(folder, dict(config, qwen3=str(_checkpoint(tmp_path, "qwen3-new.safetensors"))), entrypoint=ANIMA)
    assert _caches(folder, ANIMA_SUFFIX) == set()
    assert len(_caches(folder, SDXL_SUFFIX)) == 3  # another route's cache stays
    _build_caches(folder, ANIMA_SUFFIX)
    # Anima's cache stores the caption dropout rate.
    _refresh(
        folder,
        dict(config, caption_dropout_rate=0.1, qwen3=str(tmp_path / "models" / "qwen3-new.safetensors")),
        entrypoint=ANIMA,
    )
    assert _caches(folder, ANIMA_SUFFIX) == set()


def test_each_route_keeps_its_own_record(tmp_path):
    folder = _folder(tmp_path)
    sdxl = _train_config(_checkpoint(tmp_path))
    anima = {"cache_text_encoder_outputs_to_disk": True, "qwen3": str(_checkpoint(tmp_path, "qwen3.safetensors"))}
    _refresh(folder, sdxl)
    _refresh(folder, anima, entrypoint=ANIMA)
    _build_caches(folder, SDXL_SUFFIX)
    _refresh(folder, sdxl)
    assert len(_caches(folder)) == 3


def test_several_folders_are_handled_on_their_own(tmp_path):
    first = _folder(tmp_path, "first")
    second = _folder(tmp_path, "second")
    config = _train_config(_checkpoint(tmp_path))

    def two_folders():
        one = dataset.build_dataset_config(first)
        two = dataset.build_dataset_config(second)
        one["datasets"].append(two["datasets"][0])
        return one

    dataset.refresh_text_caches(two_folders(), config, SDXL)
    _build_caches(first)
    _build_caches(second)
    (second / "a.txt").write_text("changed", encoding="utf-8")
    dataset.refresh_text_caches(two_folders(), config, SDXL)
    assert len(_caches(first)) == 3
    assert _caches(second) == set()
    # Two subsets of one folder share its record.
    shared = dataset.build_dataset_config(first)
    shared["datasets"][0]["subsets"].append(dict(shared["datasets"][0]["subsets"][0], caption_prefix="zkx,"))
    dataset.refresh_text_caches(shared, config, SDXL)
    assert _caches(first) == set()
    _build_caches(first)
    dataset.refresh_text_caches(shared, config, SDXL)
    assert len(_caches(first)) == 3


def test_missing_folder_is_left_to_the_backend(tmp_path):
    config = _train_config(_checkpoint(tmp_path))
    dataset.refresh_text_caches(dataset.build_dataset_config(tmp_path / "gone"), config, SDXL)
    assert not (tmp_path / "gone").exists()


def _launch_plan(route_key, folder, train_config, tmp_path) -> train.LaunchPlan:
    return train.LaunchPlan(
        route_key=route_key,
        base_key="x",
        command=[sys.executable, "-c", "print('ok')"],
        config_file=tmp_path / "t.toml",
        dataset_config_file=tmp_path / "d.toml",
        sample_prompts_file=tmp_path / "p.txt",
        output_dir=tmp_path,
        estimated_steps_per_epoch=1,
        estimated_total_steps=1,
        train_config=train_config,
        dataset_config=dataset.build_dataset_config(folder),
    )


def test_execute_refreshes_before_launch(tmp_path):
    folder = _folder(tmp_path)
    _build_caches(folder)
    config = _train_config(_checkpoint(tmp_path))
    train.execute(_launch_plan("illustrious", folder, config, tmp_path), sd_scripts_dir=tmp_path)
    assert _caches(folder) == set()
    assert (folder / dataset.TEXT_CACHE_RECORD).is_file()
    # SD1.5 caches no text-encoder outputs: execute leaves the folder alone.
    sd15 = _folder(tmp_path, "sd15")
    _build_caches(sd15)
    train.execute(_launch_plan("sd15", sd15, config, tmp_path), sd_scripts_dir=tmp_path)
    assert len(_caches(sd15)) == 3


def test_each_numbered_subset_keeps_its_own_cache_record(tmp_path):
    """A folder name used as class_tokens is part of what the cache encodes."""
    root = tmp_path / "set"
    root.mkdir()
    mychar = _folder(root, "10_mychar")
    other = _folder(root, "3_other")
    checkpoint = _checkpoint(tmp_path)
    config = dataset.build_dataset_config(root)
    assert len(config["datasets"][0]["subsets"]) == 2
    dataset.refresh_text_caches(config, _train_config(checkpoint), SDXL)
    for folder in (mychar, other):
        _build_caches(folder)
    dataset.refresh_text_caches(config, _train_config(checkpoint), SDXL)
    assert len(_caches(mychar)) == 3 and len(_caches(other)) == 3  # nothing changed
    # A trigger word drops the folder names, so both caches are made again.
    with_trigger = dataset.build_dataset_config(root, caption_prefix="zkx,", folder_class_tokens=False)
    dataset.refresh_text_caches(with_trigger, _train_config(checkpoint), SDXL)
    assert _caches(mychar) == set() and _caches(other) == set()


def test_moss_image_text_cache_follows_the_text_encoder_file(tmp_path):
    """MOSS-Image's own cache check compares only the caption texts, but its bf16 and ConvRot
    INT8 text encoders encode a little differently."""
    moss = "moss_image_train_network.py"
    suffix = dataset.TEXT_CACHE_SUFFIXES[moss]
    assert suffix == "_moss_te.npz"
    folder = _folder(tmp_path)
    bf16 = _checkpoint(tmp_path, "qwen3.5_9b_bf16.safetensors", b"bf16 weights")
    int8 = _checkpoint(tmp_path, "qwen3.5_9b_int8_convrot.safetensors", b"int8")
    config = {"cache_text_encoder_outputs_to_disk": True, "text_encoder": str(bf16), "mixed_precision": "bf16"}
    _refresh(folder, config, entrypoint=moss)
    _build_caches(folder, suffix)
    _refresh(folder, dict(config, pretrained_model_name_or_path="/models/other-dit.safetensors"), entrypoint=moss)
    assert len(_caches(folder, suffix)) == 3
    _refresh(folder, dict(config, text_encoder=str(int8)), entrypoint=moss)
    assert _caches(folder, suffix) == set()
    _build_caches(folder, suffix)
    # Quantizing the bf16 file while it loads changes the outputs too.
    config = dict(config, text_encoder=str(int8), convrot_int8_text_encoder=True)
    _refresh(folder, config, entrypoint=moss)
    assert _caches(folder, suffix) == set()
    _build_caches(folder, suffix)
    # The file is known by its name, size and bytes, not its folder.
    moved = tmp_path / "fresh" / int8.name
    moved.parent.mkdir()
    moved.write_bytes(int8.read_bytes())
    _refresh(folder, dict(config, text_encoder=str(moved)), entrypoint=moss)
    assert len(_caches(folder, suffix)) == 3
    moved.write_bytes(b"INT8")
    _refresh(folder, dict(config, text_encoder=str(moved)), entrypoint=moss)
    assert _caches(folder, suffix) == set()
    _build_caches(folder, suffix)
    # Each MOSS-Image cache file stores its caption dropout rate, and training reads it from there.
    config = dict(config, text_encoder=str(moved), caption_dropout_rate=0.1)
    _refresh(folder, config, entrypoint=moss)
    _build_caches(folder, suffix)
    _refresh(folder, dict(config, caption_dropout_rate=0.0), entrypoint=moss)
    assert _caches(folder, suffix) == set()

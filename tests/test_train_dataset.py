"""Datasets and captions: the trigger word, folders and repeats, caption checks and dataset settings."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kohya_trainer import config, registry, train

from conftest import ROUTE, SD_SCRIPTS, _anima_plan, _plan, _sd15_plan, needs_submodule


@needs_submodule
def test_trigger_rides_caption_prefix_only(image_dataset, tmp_path):
    """With class_tokens too, uncaptioned images get the trigger twice: sd-scripts adds the prefix to that fallback."""
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx style")
    [subset] = plan.dataset_config["datasets"][0]["subsets"]
    assert subset["caption_prefix"] == "zkx style,"  # sd-scripts adds the joining space
    assert "class_tokens" not in subset
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("zkx style, ")


@needs_submodule
def test_trigger_keeps_text_encoder_caching(image_dataset, tmp_path, capsys):
    """An old disk cache is removed before launch when the trigger changes, so caching stays on."""
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx")
    assert plan.train_config["cache_text_encoder_outputs"] is True
    assert plan.train_config["cache_text_encoder_outputs_to_disk"] is True
    assert "caching is off" not in capsys.readouterr().out
    # A memory-only cache with a trigger is fine too.
    plan = _plan(
        image_dataset,
        tmp_path,
        trigger_word="zkx",
        config_overrides={"cache_text_encoder_outputs": True, "cache_text_encoder_outputs_to_disk": None},
    )
    assert plan.train_config["cache_text_encoder_outputs"] is True
    assert "cache_text_encoder_outputs_to_disk" not in plan.train_config


@needs_submodule
def test_trigger_joins_a_caption_prefix_override(image_dataset, tmp_path, capsys):
    """sd-scripts joins caption_prefix and the caption with one space, so the
    trigger goes first and the override's prefix follows it."""
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx", dataset_overrides={"caption_prefix": "masterpiece,"})
    [subset] = plan.dataset_config["datasets"][0]["subsets"]
    assert subset["caption_prefix"] == "zkx, masterpiece,"
    assert "'zkx, masterpiece,'" in capsys.readouterr().out
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("zkx, ")
    # A prefix that already starts with the trigger is kept as written.
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx", dataset_overrides={"caption_prefix": "zkx, solo,"})
    assert plan.dataset_config["datasets"][0]["subsets"][0]["caption_prefix"] == "zkx, solo,"
    assert "joined" not in capsys.readouterr().out
    # Without a trigger the override is used as is.
    plan = _plan(image_dataset, tmp_path, dataset_overrides={"caption_prefix": "masterpiece,"})
    assert plan.dataset_config["datasets"][0]["subsets"][0]["caption_prefix"] == "masterpiece,"


@needs_submodule
@pytest.mark.parametrize(
    "trigger, prefix, expected",
    [
        ("zkx", "zkx", "zkx,"),  # the bare trigger keeps its comma
        (" zkx ", "zkx, solo,", "zkx, solo,"),  # a stray space in the form
        ("zkx, 1girl", "zkx, 1girl, solo,", "zkx, 1girl, solo,"),  # a trigger with a comma
        ("zkx", "zkx solo", "zkx solo"),
        ("zkx", "zkxy,", "zkx, zkxy,"),  # only a whole trigger counts
    ],
)
def test_trigger_is_never_doubled_by_the_prefix(image_dataset, tmp_path, trigger, prefix, expected):
    plan = _plan(image_dataset, tmp_path, trigger_word=trigger, dataset_overrides={"caption_prefix": prefix})
    assert plan.dataset_config["datasets"][0]["subsets"][0]["caption_prefix"] == expected


@needs_submodule
def test_trigger_joins_a_caption_prefix_from_the_training_settings(image_dataset, tmp_path, capsys):
    # In the backend the subset's trigger prefix hides this one.
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx", config_overrides={"caption_prefix": "masterpiece,"})
    assert plan.dataset_config["datasets"][0]["subsets"][0]["caption_prefix"] == "zkx, masterpiece,"
    assert "caption_prefix" not in plan.train_config
    assert "joined" in capsys.readouterr().out


@needs_submodule
def test_build_plan_guards_empty_captions_directly(tmp_path):
    """The advanced notebook calls build_plan directly, so the guards live there too."""
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "a.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="empty captions"):
        _plan(bare, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        _plan(empty, tmp_path)


@needs_submodule
def test_auto_repeats_applied_when_unset(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path)  # 3 images -> clamped to 20
    assert plan.dataset_config["datasets"][0]["subsets"][0]["num_repeats"] == 20
    plan = _plan(image_dataset, tmp_path, num_repeats=7)
    assert plan.dataset_config["datasets"][0]["subsets"][0]["num_repeats"] == 7


@needs_submodule
def test_caption_dropout_with_text_caching_on_anima_only(image_dataset, tmp_path):
    plan = _anima_plan(image_dataset, tmp_path, config_overrides={"caption_dropout_rate": 0.1})
    assert plan.train_config["caption_dropout_rate"] == 0.1
    assert plan.train_config["cache_text_encoder_outputs"] is True
    with pytest.raises(ValueError, match="caption_dropout_rate"):
        _plan(image_dataset, tmp_path, config_overrides={"caption_dropout_rate": 0.1})


@needs_submodule
def test_route_repeats_target_scales_auto_repeats(tmp_path):
    """The SD1.5 route's higher learning rate pairs with fewer steps: about 300 images per epoch."""
    data = tmp_path / "many"
    data.mkdir()
    for i in range(60):
        (data / f"i{i}.png").write_bytes(b"x")
        (data / f"i{i}.txt").write_text("1girl", encoding="utf-8")
    sd15 = registry.get("sd15")
    plan = train.build_plan(
        sd15,
        sd15.base(),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=data,
        workspace=tmp_path / "ws15",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    assert plan.dataset_config["datasets"][0]["subsets"][0]["num_repeats"] == 5  # round(300/60)
    plan = _plan(data, tmp_path)  # illustrious keeps the 400 target
    assert plan.dataset_config["datasets"][0]["subsets"][0]["num_repeats"] == 7  # round(400/60)


@needs_submodule
def test_quick_train_refuses_fully_uncaptioned_without_trigger(tmp_path, fake_download):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "a.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="empty captions"):
        train.quick_train(
            model="illustrious",
            dataset=bare,
            project_name="bare",
            workspace=tmp_path / "runs",
            auto_tag=False,
            dry_run=True,
        )
    # A trigger word makes the same dataset trainable (captions = the trigger).
    plan = train.quick_train(
        model="illustrious",
        dataset=bare,
        project_name="bare",
        workspace=tmp_path / "runs",
        trigger_word="zkx",
        auto_tag=False,
        dry_run=True,
    )
    assert plan.dataset_config["datasets"][0]["subsets"][0]["caption_prefix"] == "zkx,"


@needs_submodule
def test_quick_train_nested_images_get_a_hint(tmp_path, fake_download):
    nested = tmp_path / "nested" / "concept"  # not named like 10_concept
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="subfolders"):
        train.quick_train(
            model="illustrious",
            dataset=tmp_path / "nested",
            project_name="nested",
            workspace=tmp_path / "runs",
            auto_tag=False,
            dry_run=True,
        )


@needs_submodule
def test_route_resolution_and_dataset_defaults(image_dataset, tmp_path):
    sd15 = registry.get("sd15")
    plan = train.build_plan(
        sd15,
        sd15.base(),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws15",
        project_name="test-sd15",
        sd_scripts_dir=SD_SCRIPTS,
    )
    general = plan.dataset_config["general"]
    assert general["resolution"] == 512  # route default, not the SDXL 1024
    assert general["shuffle_caption"] is True  # SD1.5 classic convention
    assert general["keep_tokens"] == 1
    assert "--w 512" in plan.sample_prompts_file.read_text(encoding="utf-8")
    assert plan.command[3].endswith("train_network.py")
    # SDXL route is unaffected: 1024, no shuffle (TE-output caching).
    sdxl_plan = _plan(image_dataset, tmp_path)
    assert sdxl_plan.dataset_config["general"]["resolution"] == 1024
    assert sdxl_plan.dataset_config["general"]["shuffle_caption"] is False


@needs_submodule
def test_sd15_trigger_composes_with_shuffle(image_dataset, tmp_path):
    """caption_prefix applies before the shuffle and keep_tokens=1 anchors it. SD1.5 has no text caching."""
    sd15 = registry.get("sd15")
    plan = train.build_plan(
        sd15,
        sd15.base(),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "ws15",
        project_name="test-sd15",
        trigger_word="zkx",
        sd_scripts_dir=SD_SCRIPTS,
    )
    [subset] = plan.dataset_config["datasets"][0]["subsets"]
    assert subset["caption_prefix"] == "zkx,"
    assert plan.dataset_config["general"]["shuffle_caption"] is True
    assert "cache_text_encoder_outputs" not in plan.train_config


@needs_submodule
def test_shuffle_with_trigger_anchors_keep_tokens(image_dataset, tmp_path):
    """A shuffle moves the trigger. keep_tokens=1 keeps it first, unless the user sets keep_tokens."""
    # Shuffling needs text caching off on this route.
    caching_off = {"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None}
    plan = _plan(
        image_dataset,
        tmp_path,
        trigger_word="zkx",
        config_overrides=dict(caching_off),
        dataset_overrides={"shuffle_caption": True},
    )
    assert plan.dataset_config["general"]["keep_tokens"] == 1
    # A keep_tokens the user sets, even 0, is theirs.
    plan = _plan(
        image_dataset,
        tmp_path,
        trigger_word="zkx",
        config_overrides=dict(caching_off),
        dataset_overrides={"shuffle_caption": True, "keep_tokens": 0},
    )
    assert plan.dataset_config["general"]["keep_tokens"] == 0
    # No shuffle: the trigger stays first anyway.
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx")
    assert plan.dataset_config["general"]["keep_tokens"] == 0


@needs_submodule
def test_drive_folder_dataset_gets_the_io_note(image_dataset, tmp_path, capsys, monkeypatch):
    """A Drive folder trains and caches latents in place on Drive, so the plan says so before caching."""
    monkeypatch.setattr(
        train.dataset_mod,
        "summarize",
        lambda d, ext=".txt": {"images": 3, "captioned": 3, "uncaptioned": 0, "nested_images": 0},
    )
    real_counts = train.dataset_mod.subset_counts
    monkeypatch.setattr(
        train.dataset_mod,
        "subset_counts",
        lambda d, ext=".txt": [{**c, "captioned": c["images"]} for c in real_counts(image_dataset, ext)],
    )
    _plan(image_dataset, tmp_path, dataset_dir="/content/drive/MyDrive/data")
    assert "Google Drive" in capsys.readouterr().out
    _plan(image_dataset, tmp_path)
    assert "Google Drive" not in capsys.readouterr().out


def _numbered(root, captions=True):
    """10_mychar (2 images), 3_red_hair (1 image) and one loose image on top."""
    for folder, names in ((root / "10_mychar", ("a", "b")), (root / "3_red_hair", ("c",)), (root, ("top",))):
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / f"{name}.png").write_bytes(b"x")
            if captions:
                (folder / f"{name}.txt").write_text("1girl", encoding="utf-8")
    return root


@needs_submodule
def test_numbered_folders_plan_one_subset_each_and_sum_the_estimate(tmp_path, capsys):
    root = _numbered(tmp_path / "data")
    plan = _plan(root, tmp_path, num_repeats=4, train_batch_size=1)
    subsets = {Path(s["image_dir"]).name: s for s in plan.dataset_config["datasets"][0]["subsets"]}
    assert {name: s["num_repeats"] for name, s in subsets.items()} == {"data": 4, "10_mychar": 10, "3_red_hair": 3}
    assert subsets["10_mychar"]["class_tokens"] == "mychar"
    assert plan.estimated_steps_per_epoch == 2 * 10 + 1 * 3 + 1 * 4
    out = capsys.readouterr().out
    assert "the folder numbers are used" in out and "top folder" in out
    assert "10_mychar 2 x 10 + 3_red_hair 1 x 3" in out


@needs_submodule
def test_a_trigger_keeps_folder_names_out_of_the_captions(tmp_path):
    plan = _plan(_numbered(tmp_path / "data"), tmp_path, trigger_word="zkx")
    for subset in plan.dataset_config["datasets"][0]["subsets"]:
        assert "class_tokens" not in subset
        assert subset["caption_prefix"] == "zkx,"


@needs_submodule
def test_folder_names_count_as_captions(tmp_path):
    root = tmp_path / "data" / "10_mychar"
    root.mkdir(parents=True)
    (root / "a.png").write_bytes(b"x")
    plan = _plan(tmp_path / "data", tmp_path)  # no caption file and no trigger
    [subset] = plan.dataset_config["datasets"][0]["subsets"]
    assert subset["class_tokens"] == "mychar"


@needs_submodule
def test_auto_repeats_count_only_the_images_without_a_folder_number(tmp_path):
    root = _numbered(tmp_path / "data")
    plan = _plan(root, tmp_path)
    top = next(s for s in plan.dataset_config["datasets"][0]["subsets"] if Path(s["image_dir"]).name == "data")
    assert top["num_repeats"] == train.dataset_mod.default_num_repeats(1, ROUTE.repeats_target)


@needs_submodule
def test_quick_train_auto_tags_only_when_no_subset_has_captions(tmp_path, fake_download, monkeypatch):
    tagged = []
    monkeypatch.setattr(train.tagging, "tag_images", lambda folder, **kw: tagged.append(folder))
    root = tmp_path / "data"
    (root / "10_mychar").mkdir(parents=True)
    (root / "10_mychar" / "a.png").write_bytes(b"x")
    (root / "10_mychar" / "a.txt").write_text("1girl", encoding="utf-8")
    (root / "b.png").write_bytes(b"x")
    common = dict(model="illustrious", project_name="p", workspace=tmp_path / "runs", dry_run=True, trigger_word="z")
    train.quick_train(dataset=root, **common)
    assert tagged == []  # the numbered folder has a caption, so nothing is replaced
    (root / "10_mychar" / "a.txt").unlink()
    train.quick_train(dataset=root, **common)
    assert tagged == [root.resolve()]


@needs_submodule
def test_quick_train_folds_hair_and_eye_tags_into_the_trigger_only_after_its_own_tagging(
    tmp_path, fake_download, monkeypatch
):
    def fake_tagger(folder, caption_extension=".txt", **kw):
        for image in Path(folder).glob("*.png"):
            image.with_suffix(caption_extension).write_text("1girl, long hair, green eyes, sundress", encoding="utf-8")

    monkeypatch.setattr(train.tagging, "tag_images", fake_tagger)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("a", "b"):
        (data / f"{name}.png").write_bytes(b"x")
    common = dict(model="illustrious", dataset=data, workspace=tmp_path / "runs", dry_run=True, project_name="p")

    def captions():
        texts = {p.read_text(encoding="utf-8") for p in data.glob("*.txt")}
        for caption in data.glob("*.txt"):
            caption.unlink()  # so the next run tags again
        return texts

    plan = train.quick_train(trigger_word="zkx", **common)
    assert captions() == {"1girl, sundress"}
    # The preview prompt is built from the pruned captions.
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("zkx, 1girl, sundress, cowboy shot")
    train.quick_train(trigger_word="zkx", prune_appearance_tags=False, **common)
    assert captions() == {"1girl, long hair, green eyes, sundress"}
    # Without a trigger nothing learns the look, so the tags stay.
    train.quick_train(**common)
    assert captions() == {"1girl, long hair, green eyes, sundress"}
    # Captions the user brought are never changed.
    for name in ("a", "b"):
        (data / f"{name}.txt").write_text("1girl, long hair, green eyes", encoding="utf-8")
    train.quick_train(trigger_word="zkx", **common)
    assert captions() == {"1girl, long hair, green eyes"}
    # Another caption extension reaches the pruning and the preview prompt.
    plan = train.quick_train(trigger_word="zkx", dataset_overrides={"caption_extension": ".caption"}, **common)
    assert {p.read_text(encoding="utf-8") for p in data.glob("*.caption")} == {"1girl, sundress"}
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("zkx, 1girl, sundress, cowboy shot")


@needs_submodule
def test_quick_train_follows_a_caption_extension_override(tmp_path, fake_download, monkeypatch):
    tagged = []
    monkeypatch.setattr(train.tagging, "tag_images", lambda folder, **kw: tagged.append(kw))
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.png").write_bytes(b"x")
    (data / "a.caption").write_text("1girl", encoding="utf-8")
    common = dict(model="illustrious", dataset=data, workspace=tmp_path / "runs", dry_run=True)
    plan = train.quick_train(project_name="p", dataset_overrides={"caption_extension": ".caption"}, **common)
    assert tagged == []  # the .caption file counts, so nothing is tagged
    assert plan.dataset_config["general"]["caption_extension"] == ".caption"
    (data / "a.caption").unlink()
    train.quick_train(project_name="q", dataset_overrides={"caption_extension": ".caption"}, trigger_word="z", **common)
    assert tagged == [{"caption_extension": ".caption"}]  # the tagger writes what the trainer reads


@needs_submodule
def test_txt_captions_under_another_extension_get_a_note(image_dataset, tmp_path, capsys):
    _plan(image_dataset, tmp_path, trigger_word="z", dataset_overrides={"caption_extension": ".caption"})
    out = capsys.readouterr().out
    assert "no image has a .caption caption file, but 2 have a .txt one" in out
    _plan(image_dataset, tmp_path)
    assert "caption file, but" not in capsys.readouterr().out


@needs_submodule
def test_dataset_keys_in_config_overrides_move_to_the_dataset_config(image_dataset, tmp_path, capsys):
    plan = _plan(
        image_dataset,
        tmp_path,
        config_overrides={"flip_aug": True, "max_bucket_reso": 1536, "train_batch_size": 3, "resolution": "768,1024"},
        dataset_overrides={"max_bucket_reso": 1280},
    )
    general = plan.dataset_config["general"]
    assert general["flip_aug"] is True
    assert general["max_bucket_reso"] == 1280  # dataset_overrides wins
    assert general["resolution"] == [768, 1024]
    assert plan.dataset_config["datasets"][0]["batch_size"] == 3
    for key in ("flip_aug", "max_bucket_reso", "train_batch_size", "resolution"):
        assert key not in plan.train_config
    out = capsys.readouterr().out
    assert "moved to the dataset settings" in out and "dataset_overrides also sets max_bucket_reso" in out
    # Subset settings the dataset config does not write keep the command-line route.
    plan = _sd15_plan(image_dataset, tmp_path, config_overrides={"caption_dropout_rate": 0.1, "color_aug": True})
    assert plan.train_config["caption_dropout_rate"] == 0.1 and plan.train_config["color_aug"] is True


@needs_submodule
@pytest.mark.parametrize("key", ["train_data_dir", "reg_data_dir", "in_json"])
def test_dataset_arguments_the_backend_ignores_are_refused(image_dataset, tmp_path, fake_download, monkeypatch, key):
    with pytest.raises(ValueError, match="the trainer ignores") as error:
        _plan(image_dataset, tmp_path, config_overrides={key: "/content/drive/MyDrive/reg"})
    assert "10_name" in str(error.value)
    assert not (tmp_path / "ws").exists()  # refused before anything is written

    def no_download(*args, **kwargs):
        raise AssertionError("downloaded before refusing")

    monkeypatch.setattr(train.models, "prepare_base", no_download)
    with pytest.raises(ValueError, match="the trainer ignores"):
        train.quick_train(
            model="sdxl",
            dataset=image_dataset,
            project_name="p",
            workspace=tmp_path / "runs",
            config_overrides={key: "x"},
        )
    # None removes a key, as everywhere in config_overrides.
    _plan(image_dataset, tmp_path, config_overrides={key: None})


@needs_submodule
def test_ignored_dataset_arguments_match_the_pin():
    source = (SD_SCRIPTS / "train_network.py").read_text(encoding="utf-8")
    assert 'ignored = ["train_data_dir", "reg_data_dir", "in_json"]' in source
    mapping = (SD_SCRIPTS / "library" / "config_util.py").read_text(encoding="utf-8")
    assert '"train_batch_size": "batch_size",' in mapping and '"dataset_repeats": "num_repeats",' in mapping


@needs_submodule
def test_dataset_overrides_keys_are_sorted_or_refused(image_dataset, tmp_path, capsys):
    """build_dataset_config takes only its own keyword arguments. Other
    dataset settings reach the trainer through config_overrides."""
    plan = _sd15_plan(
        image_dataset,
        tmp_path,
        dataset_overrides={"caption_dropout_rate": 0.1, "color_aug": True, "train_batch_size": 1, "flip_aug": True},
    )
    assert plan.train_config["caption_dropout_rate"] == 0.1 and plan.train_config["color_aug"] is True
    assert plan.dataset_config["datasets"][0]["batch_size"] == 1
    assert plan.dataset_config["general"]["flip_aug"] is True
    out = capsys.readouterr().out
    assert "caption_dropout_rate, color_aug in dataset_overrides moved to config_overrides" in out
    assert "train_batch_size in dataset_overrides is batch_size" in out
    # The same key in config_overrides is the user's too, and wins.
    plan = _sd15_plan(
        image_dataset,
        tmp_path,
        dataset_overrides={"caption_dropout_rate": 0.1},
        config_overrides={"caption_dropout_rate": 0.2},
    )
    assert plan.train_config["caption_dropout_rate"] == 0.2
    with pytest.raises(ValueError, match="dataset_overrides cannot take learning_rate. It takes batch_size") as error:
        _plan(image_dataset, tmp_path / "x", dataset_overrides={"learning_rate": 1e-4})
    assert "class_tokens" in str(error.value)
    assert not (tmp_path / "x").exists()  # refused before anything is written


@needs_submodule
def test_quick_train_checks_dataset_overrides_before_downloading(image_dataset, tmp_path, monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("downloaded before refusing")

    monkeypatch.setattr(train.models, "prepare_base", no_download)
    with pytest.raises(ValueError, match="dataset_overrides cannot take random_crops"):
        train.quick_train(
            model="sdxl",
            dataset=image_dataset,
            project_name="p",
            workspace=tmp_path / "runs",
            dataset_overrides={"random_crops": True},
        )


@needs_submodule
def test_dataset_fallback_args_match_the_pin():
    """Each key moved from dataset_overrides falls back to the training arguments in the pinned
    dataset schema, and is a training argument on every route."""
    source = (SD_SCRIPTS / "library" / "config_util.py").read_text(encoding="utf-8")
    schema = source[source.index("SUBSET_ASCENDABLE_SCHEMA = {") : source.index("ARGPARSE_SPECIFIC_SCHEMA")]
    assert "generate_params_by_fallbacks" in source and "argparse_config, runtime_params]" in source
    for key in train.DATASET_FALLBACK_ARGS:
        assert f'"{key}"' in schema, key
    assert not train.DATASET_FALLBACK_ARGS & train.dataset_override_keys()
    for route in registry.ROUTES.values():
        config.validate_keys(dict.fromkeys(train.DATASET_FALLBACK_ARGS), SD_SCRIPTS, route.entrypoint)


@needs_submodule
def test_top_folder_images_with_nothing_to_train_on_get_a_note(tmp_path, fake_download, capsys):
    """Images next to numbered folders get no folder name, so without a caption file,
    a trigger or class_tokens they train on an empty caption."""
    root = _numbered(tmp_path / "data", captions=False)
    (root / "top2.png").write_bytes(b"x")
    plan = _plan(root, tmp_path)
    assert len(plan.dataset_config["datasets"][0]["subsets"]) == 3
    assert "note: 2 images have no caption file and nothing else to train on" in capsys.readouterr().out
    plan = train.quick_train(
        model="illustrious", dataset=root, project_name="p", workspace=tmp_path / "runs", auto_tag=False, dry_run=True
    )
    assert "2 images have no caption file and nothing else" in capsys.readouterr().out
    # class_tokens or a trigger covers them.
    for extra in (dict(dataset_overrides={"class_tokens": "mychar"}), dict(trigger_word="zkx")):
        _plan(root, tmp_path, **extra)
        assert "nothing else to train on" not in capsys.readouterr().out
    # A caption file elsewhere does not hide them either.
    (root / "10_mychar" / "a.txt").write_text("1girl", encoding="utf-8")
    _plan(root, tmp_path)
    assert "2 images have no caption file and nothing else" in capsys.readouterr().out
    # A caption_prefix or caption_suffix is a caption too: the backend adds it to an empty one.
    (root / "10_mychar" / "a.txt").unlink()
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "a.png").write_bytes(b"x")
    for extra in (
        dict(dataset_overrides={"caption_prefix": "mychar"}),
        dict(config_overrides={"caption_suffix": "mychar"}),
        dict(dataset_overrides={"caption_suffix": "mychar"}),
    ):
        _plan(bare, tmp_path, **extra)
        assert "nothing else to train on" not in capsys.readouterr().out
        train.quick_train(
            model="illustrious",
            dataset=bare,
            project_name="p",
            workspace=tmp_path / "runs",
            auto_tag=False,
            dry_run=True,
            **extra,
        )
    with pytest.raises(ValueError, match="empty captions"):
        _plan(bare, tmp_path)
    (root / "10_mychar" / "a.txt").write_text("1girl", encoding="utf-8")
    # In 1-click, auto_tag left them alone because one image has a caption: say so.
    common = dict(model="illustrious", dataset=root, project_name="p", workspace=tmp_path / "runs", dry_run=True)
    train.quick_train(auto_tag=True, **common)
    assert "auto_tag did not run, because some images have caption files" in capsys.readouterr().out
    train.quick_train(auto_tag=False, **common)
    assert "auto_tag did not run" not in capsys.readouterr().out


@needs_submodule
def test_quick_train_counts_class_tokens_as_build_plan_does(tmp_path, fake_download):
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "a.png").write_bytes(b"x")
    common = dict(model="illustrious", dataset=bare, workspace=tmp_path / "runs", auto_tag=False, dry_run=True)
    with pytest.raises(ValueError, match="empty captions"):
        train.quick_train(project_name="p", **common)
    plan = train.quick_train(project_name="p", dataset_overrides={"class_tokens": "mychar"}, **common)
    assert plan.dataset_config["datasets"][0]["subsets"][0]["class_tokens"] == "mychar"
    # A folder name counts the same way in both.
    (tmp_path / "named" / "10_mychar").mkdir(parents=True)
    (tmp_path / "named" / "10_mychar" / "a.png").write_bytes(b"x")
    train.quick_train(project_name="q", **{**common, "dataset": tmp_path / "named"})


@needs_submodule
def test_several_caption_candidates_are_refused_where_the_text_cache_keeps_one(image_dataset, tmp_path, monkeypatch):
    """Anima's trainer stops after it loads the dataset when a cached caption has
    several candidates. A caption with line breaks but no separator line is fine."""
    multiline = {"enable_multiline_caption": True}
    (image_dataset / "a.txt").write_text("a sentence.\nanother line.", encoding="utf-8")
    _anima_plan(image_dataset, tmp_path, config_overrides=multiline)
    (image_dataset / "a.txt").write_text("1girl, solo\n---===---\na girl alone.", encoding="utf-8")
    with pytest.raises(ValueError, match=r"such as the one for a\.png.*keeps only one caption per image"):
        _anima_plan(image_dataset, tmp_path, config_overrides=multiline)
    with pytest.raises(ValueError, match="several captions"):
        _anima_plan(image_dataset, tmp_path, dataset_overrides=multiline)  # moved to the training settings
    # Without multiline captions the backend reads the first line only, and
    # without text caching each step encodes the candidate it picks.
    _anima_plan(image_dataset, tmp_path)
    no_cache = {"cache_text_encoder_outputs": None, "cache_text_encoder_outputs_to_disk": None}
    _anima_plan(image_dataset, tmp_path, config_overrides={**multiline, **no_cache})

    # quick_train refuses a folder on disk before any download.
    def no_download(*args, **kw):
        raise AssertionError("downloaded before refusing")

    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(SD_SCRIPTS))
    monkeypatch.setattr(train.models, "prepare_base", no_download)
    with pytest.raises(ValueError, match="several captions"):
        train.quick_train(
            model="anima",
            dataset=str(image_dataset),
            project_name="p",
            workspace=tmp_path / "runs",
            config_overrides=multiline,
        )


def test_route_dataset_defaults_go_where_the_backend_reads_them():
    params = train.dataset_override_keys()
    for route in registry.ROUTES.values():
        for key, _ in route.dataset_defaults:
            assert key in params or key in train.DATASET_FALLBACK_ARGS, (route.key, key)


@needs_submodule
def test_quick_train_tags_by_the_routes_default(tmp_path, fake_download, monkeypatch):
    """MOSS-Image leaves a set with no captions to the trigger word. The tag routes tag it."""
    tagged = []
    monkeypatch.setattr(train.tagging, "tag_images", lambda folder, **kw: tagged.append(folder))
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.png").write_bytes(b"x")
    common = dict(dataset=data, workspace=tmp_path / "runs", dry_run=True)
    plan = train.quick_train(model="moss_image", project_name="m", trigger_word="zkx", convrot_int8="off", **common)
    assert tagged == []
    assert 'caption_prefix = "zkx,"' in plan.dataset_config_file.read_text(encoding="utf-8")
    train.quick_train(
        model="moss_image", project_name="m2", trigger_word="zkx", auto_tag=True, convrot_int8="off", **common
    )
    assert tagged == [data.resolve()]
    train.quick_train(model="illustrious", project_name="i", trigger_word="zkx", **common)
    assert len(tagged) == 2
    # With neither a trigger nor captions, it stops before any model downloads.
    downloads = []
    monkeypatch.setattr(train.models, "prepare_base", lambda *args, **kwargs: downloads.append(args))
    with pytest.raises(ValueError, match="no captions and no trigger_word"):
        train.quick_train(model="moss_image", project_name="m3", convrot_int8="off", **common)
    # A .zip too: it is unpacked and counted first.
    monkeypatch.setattr(train.environment, "scratch_dir", lambda name: tmp_path / "scratch" / name)
    archive = tmp_path / "set.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(data / "a.png", "a.png")
    with pytest.raises(ValueError, match="no captions and no trigger_word"):
        train.quick_train(
            model="moss_image", project_name="m4", dataset=archive, workspace=tmp_path / "runs", dry_run=True
        )
    assert downloads == []

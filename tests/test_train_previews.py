"""The preview prompts and their sample settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from kohya_trainer import registry, train

from conftest import ROUTE, SD_SCRIPTS, _anima_plan, _plan, needs_submodule


@needs_submodule
def test_sample_params_are_route_aware(image_dataset, tmp_path):
    """Anima's model card wants CFG 4-6 at 30-50 steps. The SD routes keep the 24-step preview."""
    prompt = _plan(image_dataset, tmp_path).sample_prompts_file.read_text(encoding="utf-8")
    assert "--s 24" in prompt and "--l 6" in prompt
    prompt = _anima_plan(image_dataset, tmp_path).sample_prompts_file.read_text(encoding="utf-8")
    assert "--s 30" in prompt and "--l 5" in prompt


@needs_submodule
def test_sample_negative_prompt_lands_in_the_prompt_file(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, sample_negative_prompt="bad hands, watermark")
    assert "--n bad hands, watermark --w" in plan.sample_prompts_file.read_text(encoding="utf-8")
    plan = _plan(image_dataset, tmp_path)  # default = the Illustrious card's own negative
    assert "--n nsfw, lowres, worst quality, comic" in plan.sample_prompts_file.read_text(encoding="utf-8")


@needs_submodule
def test_sample_defaults_are_per_notebook(image_dataset, tmp_path):
    """One preview default per route, from its most-used model's card. A typed prompt is used as written."""
    plan = _plan(image_dataset, tmp_path)  # illustrious: the v0.1 card values
    text = plan.sample_prompts_file.read_text(encoding="utf-8")
    # The dataset's common tags, then the neutral scene, then the quality tags.
    assert text.startswith(f"1girl, solo, smile, {SCENE}, masterpiece, best quality --n")
    assert "--l 6" in text and "--s 24" in text
    assert plan.train_config["sample_sampler"] == "euler_a"  # preset value
    # Animagine's card: subject first, score tags last, CFG 5 at 28 steps.
    sdxl = registry.get("sdxl")
    plan = train.build_plan(
        sdxl,
        sdxl.base(),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "wsx",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    text = plan.sample_prompts_file.read_text(encoding="utf-8")
    assert text.startswith(f"1girl, solo, smile, {SCENE}, masterpiece, high score, great score, absurdres --n")
    assert "--l 5" in text and "--s 28" in text
    # A v-pred base pick swaps the preview sampler to plain euler.
    vpred_plan = train.build_plan(
        ROUTE,
        ROUTE.base("noobai-xl-vpred-1.0"),
        checkpoint_path=Path("/models/base.safetensors"),
        vae_path=None,
        dataset_dir=image_dataset,
        workspace=tmp_path / "wsv",
        project_name="test-lora",
        sd_scripts_dir=SD_SCRIPTS,
    )
    assert vpred_plan.train_config["sample_sampler"] == "euler"
    # Explicit form values beat the notebook defaults.
    plan = _plan(image_dataset, tmp_path, sample_prompt="my prompt", sample_negative_prompt="my neg")
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("my prompt --n my neg")


@needs_submodule
def test_an_empty_negative_prompt_means_no_negative(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, sample_negative_prompt="")
    text = plan.sample_prompts_file.read_text(encoding="utf-8")
    assert " --n" not in text and "--w 832" in text
    plan = _plan(image_dataset, tmp_path, sample_negative_prompt=None)
    assert "--n nsfw, lowres, worst quality, comic" in plan.sample_prompts_file.read_text(encoding="utf-8")


@needs_submodule
def test_flags_written_in_a_prompt_win_over_ours(image_dataset, tmp_path):
    plan = _plan(image_dataset, tmp_path, sample_prompt="1girl, smile --s 40 --ss dpm_2 --w 832 --h 1216")
    [line] = plan.sample_prompts_file.read_text(encoding="utf-8").splitlines()
    assert line.startswith("1girl, smile --s 40 --ss dpm_2 --w 832 --h 1216 --n ")
    assert line.count("--s ") == 1 and line.count("--w ") == 1
    assert line.endswith(" --l 6 --d 42")


@needs_submodule
def test_sample_prompt_takes_a_prompt_file(image_dataset, tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("# my previews\n1girl, smile\n\n1girl, standing --n bad hands --w 768 --d 7\n", encoding="utf-8")
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx", sample_prompt=str(prompts))
    lines = plan.sample_prompts_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# my previews"
    assert lines[1].startswith("1girl, smile --n nsfw, lowres")  # the trigger only goes in the built-in prompt
    assert lines[1].endswith("--w 832 --h 1216 --s 24 --l 6 --d 42")  # portrait where the line sets no size
    assert lines[2] == "1girl, standing --n bad hands --w 768 --d 7 --h 1216 --s 24 --l 6"
    assert len(lines) == 3
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _plan(image_dataset, tmp_path, sample_prompt=str(tmp_path / "missing.txt"))


@needs_submodule
def test_samples_and_notes_use_the_size_and_batch_that_train(image_dataset, tmp_path, capsys, monkeypatch):
    plan = _plan(image_dataset, tmp_path, resolution=1024, dataset_overrides={"resolution": 768})
    assert plan.dataset_config["general"]["resolution"] == 768
    assert "--w 640 --h 896" in plan.sample_prompts_file.read_text(encoding="utf-8")
    # A size that is not square is kept as it trains.
    plan = _plan(image_dataset, tmp_path, dataset_overrides={"resolution": [1216, 832]})
    assert "--w 1216 --h 832" in plan.sample_prompts_file.read_text(encoding="utf-8")
    monkeypatch.setattr(train.environment, "gpu_name", lambda: "Tesla T4")
    capsys.readouterr()
    # The form says batch 1, but dataset_overrides trains at 4.
    _plan(image_dataset, tmp_path, network_type="loha", train_batch_size=1, dataset_overrides={"batch_size": 4})
    assert "set train_batch_size=1 (now 4)" in capsys.readouterr().out
    _anima_plan(image_dataset, tmp_path, dataset_overrides={"resolution": 768, "batch_size": 1})
    assert "a T4 does not support natively" not in capsys.readouterr().out
    _anima_plan(image_dataset, tmp_path, resolution=768, train_batch_size=1, dataset_overrides={"resolution": 1024})
    assert "a T4 does not support natively" in capsys.readouterr().out


SCENE = "cowboy shot, looking at viewer, simple background"


@pytest.mark.parametrize(
    "key, expected",
    [
        ("sd15", f"masterpiece, best quality, zkx, solo, {SCENE}"),
        ("sdxl", f"zkx, solo, {SCENE}, masterpiece, high score, great score, absurdres"),
        ("illustrious", f"zkx, solo, {SCENE}, masterpiece, best quality"),
        ("anima", f"masterpiece, best quality, score_7, safe, zkx, solo, {SCENE}"),
    ],
)
def test_default_preview_keeps_each_route_quality_tags_in_place(key, expected):
    route = registry.get(key)
    prompt = train.default_sample_prompt(route, "zkx")
    assert prompt == expected
    for look in ("green hair", "sweater", "beanie", "night", "outdoors"):
        assert look not in prompt
    # Without a trigger the prompt is the same less the trigger word.
    assert train.default_sample_prompt(route) == expected.replace("zkx, ", "")
    # The dataset's common tags take the place of solo, and the trigger is not doubled.
    tagged = train.default_sample_prompt(route, "zkx", ["1girl", "zkx", "blue dress"])
    assert "zkx, 1girl, blue dress, cowboy shot" in tagged and tagged.count("zkx") == 1


@pytest.mark.parametrize(
    "key, tag",
    [
        ("sd15", r"genderswap \(mtf\)"),
        ("sdxl", r"genderswap \(mtf\)"),
        ("illustrious", r"genderswap \(mtf\)"),
        ("anima", "genderswap (mtf)"),
        ("moss_image", "genderswap (mtf)"),
    ],
)
def test_default_preview_escapes_brackets_where_the_trainer_reads_weights(key, tag):
    route = registry.get(key)
    prompt = train.default_sample_prompt(route, "zkx", ["1girl", "genderswap (mtf)"])
    after = "looking at the viewer" if route.sample_sentence else "cowboy shot"
    assert f"zkx, 1girl, {tag}, {after}" in prompt
    # A tag already escaped stays as it is, and so does [x] on the plain-text routes.
    prompt = train.default_sample_prompt(route, "", [r"saber \(fate\)", "[x]"])
    escaped = route.entrypoint in ("train_network.py", "sdxl_train_network.py")
    assert (r"saber \(fate\), \[x\]" if escaped else r"saber \(fate\), [x]") in prompt


@pytest.mark.parametrize("key", sorted(registry.ROUTES))
def test_default_preview_leaves_out_tags_in_the_route_negative(key):
    route = registry.get(key)
    negative = [tag.strip() for tag in route.default_sample_negative.split(",")]
    tags = ["1girl", *negative, negative[-1].upper().replace(" ", "_"), "blue dress"]
    prompt = train.default_sample_prompt(route, "zkx", tags)
    after = "looking at the viewer" if route.sample_sentence else "cowboy shot"
    assert f"zkx, 1girl, blue dress, {after}" in prompt
    assert train.default_sample_prompt(route, "zkx", negative) == train.default_sample_prompt(route, "zkx")


@pytest.mark.parametrize("key", sorted(registry.ROUTES))
def test_every_default_negative_starts_with_nsfw_and_lowres(key):
    negative = registry.get(key).default_sample_negative
    assert negative.startswith("nsfw, lowres, ")
    tags = [tag.strip() for tag in negative.split(",")]
    assert len(tags) == len(set(tags)), negative  # lowres moved, not doubled
    if key in ("sdxl", "illustrious", "anima"):
        assert "signature" in tags, key
    if key == "anima":
        assert {"signature", "watermark"} <= set(tags)


@pytest.mark.parametrize(
    "size, expected",
    [
        ((1024, 1024), (832, 1216)),
        ((512, 512), (512, 768)),
        ((768, 768), (640, 896)),
        ((640, 640), (640, 960)),
        # At 640 and below only the height is rounded.
        ((600, 600), (600, 896)),
    ],
)
def test_square_training_sizes_preview_in_portrait(size, expected):
    assert train.sample_size(*size) == expected
    assert train.sample_size(832, 1216) == (832, 1216)
    assert train.sample_size(1216, 832) == (1216, 832)


@needs_submodule
def test_trigger_word_and_common_tags_lead_the_default_preview_prompt(image_dataset, tmp_path, capsys):
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx")
    text = plan.sample_prompts_file.read_text(encoding="utf-8")
    prompt = f"zkx, 1girl, solo, smile, {SCENE}, masterpiece, best quality"
    assert text.startswith(f"{prompt} --n")
    assert f"preview prompt (832x1216): {prompt}" in capsys.readouterr().out
    # A typed prompt is used as written: no trigger, no tags, and no note.
    plan = _plan(image_dataset, tmp_path, trigger_word="zkx", sample_prompt="my prompt")
    assert plan.sample_prompts_file.read_text(encoding="utf-8").startswith("my prompt --n")
    assert "preview prompt" not in capsys.readouterr().out
    # Sentence captions give no common tags, so the subject is solo.
    for name in ("a", "b"):
        (image_dataset / f"{name}.txt").write_text("A girl with long hair stands in a quiet park.", encoding="utf-8")
    text = _plan(image_dataset, tmp_path, trigger_word="zkx").sample_prompts_file.read_text(encoding="utf-8")
    assert text.startswith(f"zkx, solo, {SCENE}, masterpiece, best quality --n")


@needs_submodule
def test_default_preview_reads_captions_as_the_trainer_does(image_dataset, tmp_path):
    # The SDXL trainer reads only the first line, so a sentence under the tags is not seen.
    for name in ("a", "b"):
        (image_dataset / f"{name}.txt").write_text("1girl, blue dress\nA girl in a blue dress.", encoding="utf-8")
    text = _plan(image_dataset, tmp_path, trigger_word="zkx").sample_prompts_file.read_text(encoding="utf-8")
    assert text.startswith(f"zkx, 1girl, blue dress, {SCENE}, masterpiece, best quality --n")
    # The tags come from the caption files the trainer reads, set from either dict.
    for name in ("a", "b"):
        (image_dataset / f"{name}.caption").write_text("1girl, red dress", encoding="utf-8")
    for overrides in (
        {"dataset_overrides": {"caption_extension": ".caption"}},
        {"config_overrides": {"caption_extension": ".caption"}},
    ):
        text = _plan(image_dataset, tmp_path, trigger_word="zkx", **overrides).sample_prompts_file.read_text(
            encoding="utf-8"
        )
        assert text.startswith(f"zkx, 1girl, red dress, {SCENE}"), overrides

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

from kohya_trainer import dataset, registry, train

from conftest import SD_SCRIPTS, needs_submodule


def test_tag_report_counts_ranks_and_scopes(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("1girl, solo, red hair", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("1girl, solo", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"x")  # uncaptioned: contributes nothing
    (tmp_path / "stray.txt").write_text("orphan tag", encoding="utf-8")  # no image: not training data
    report = dataset.tag_report(tmp_path)
    assert "3 distinct tags across 2 captioned images" in report
    assert report.index("1girl") < report.index("red hair")  # frequency order
    assert "orphan" not in report
    top_one = dataset.tag_report(tmp_path, top=1)
    assert "1girl" in top_one and "red hair" not in top_one


def test_tag_report_without_captions(tmp_path):
    assert dataset.tag_report(tmp_path) == "no caption tags to report"


def _captions(folder: Path, *captions: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, caption in enumerate(captions):
        (folder / f"{i}.png").write_bytes(b"x")
        (folder / f"{i}.txt").write_text(caption, encoding="utf-8")
    return folder


def _frequent(folder: Path, train_config: dict | None = None, **kwargs) -> list[str]:
    return dataset.frequent_tags(dataset.build_dataset_config(folder), train_config or {}, **kwargs)


def test_frequent_tags_keep_tags_in_half_the_images_with_count_tags_first(tmp_path):
    data = _captions(
        tmp_path / "data",
        "blonde_hair, solo, 1girl, school uniform, simple background, upper body, masterpiece",
        "blonde hair, 1girl, solo, school uniform, white background, looking at viewer, signature",
        "blonde hair, 1girl, solo, sundress, white background, looking at viewer, signature",
        "blonde hair, 1girl, sundress, comic, speech bubble, english text, general",
    )
    (data / "4.png").write_bytes(b"x")  # no caption file: not counted
    tags = _frequent(data)
    # blonde_hair and blonde hair are one tag. Count tags come first, then the rest by frequency.
    assert tags == ["1girl", "solo", "blonde hair", "school uniform", "sundress"]
    assert _frequent(data, min_share=0.75) == ["1girl", "solo", "blonde hair"]


def test_frequent_tags_keep_the_captions_own_spelling(tmp_path):
    data = _captions(tmp_path / "data", "1girl, blonde_hair, school_uniform", "1girl, blonde_hair, school_uniform")
    assert _frequent(data) == ["1girl", "blonde_hair", "school_uniform"]
    # The spelling most captions use wins. On a tie, the first one seen wins.
    data = _captions(tmp_path / "mixed", "blonde_hair, red_eyes", "blonde hair, red eyes", "blonde hair, red_eyes")
    assert _frequent(data) == ["blonde hair", "red_eyes"]
    data = _captions(tmp_path / "tie", "blonde_hair", "blonde hair")
    assert _frequent(data) == ["blonde_hair"]


def test_frequent_tags_leave_out_framing_background_text_quality_and_rating(tmp_path):
    left_out = (
        "cowboy shot, full body, portrait, close-up, looking to the side, grey background, gradient background, "
        "text, watermark, artist name, twitter username, dated, best quality, absurdres, score_7, safe, nsfw, "
        "rating:general, year 2024, newest"
    )
    data = _captions(tmp_path / "data", f"1girl, {left_out}", f"1girl, {left_out}")
    assert _frequent(data) == ["1girl"]
    # Aesthetic, quality, score and text tags too, by name or by their ending, in any case.
    left_out = (
        "very aesthetic, aesthetic, displeasing, very displeasing, bad quality, great quality, amazing quality, "
        "high score, great score, good score, average score, bad score, low score, japanese text, chinese text, "
        "korean text, sound effects, logo, blurry, jpeg artifacts, Very_Aesthetic, newest aesthetic, "
        "russian text, Top Score, okay quality"
    )
    data = _captions(tmp_path / "more", f"1girl, {left_out}", f"1girl, {left_out}")
    assert _frequent(data) == ["1girl"]


def test_frequent_tags_stop_at_the_cap(tmp_path):
    caption = ", ".join(f"tag{i}" for i in range(15)) + ", 1girl"
    data = _captions(tmp_path / "data", caption, caption)
    tags = _frequent(data)
    assert len(tags) == 10 and tags[0] == "1girl"
    assert _frequent(data, top=3) == ["1girl", "tag0", "tag1"]


def test_frequent_tags_are_empty_for_sentence_captions(tmp_path):
    sentence = "A girl with long blonde hair stands by the window."
    data = _captions(tmp_path / "data", sentence, f"1girl, solo\n---===---\n{sentence}", "1girl, solo")
    multiline = {"enable_multiline_caption": True}
    assert _frequent(data, multiline) == []  # two of three captions have a sentence
    data = _captions(tmp_path / "tags", "1girl, solo", "1girl, solo", sentence)
    assert _frequent(data) == ["1girl", "solo"]  # the sentence is not a tag
    assert _frequent(_captions(tmp_path / "empty")) == []


def test_frequent_tags_count_long_character_tags_as_tags(tmp_path):
    # Six words, but the bracketed one does not count toward a sentence.
    caption = "1girl, jeanne d'arc alter santa lily (fate), smile"
    data = _captions(tmp_path / "data", caption, caption)
    assert _frequent(data) == ["1girl", "jeanne d'arc alter santa lily (fate)", "smile"]
    caption = r"1girl, jeanne d'arc alter santa lily \(fate\), smile"
    data = _captions(tmp_path / "escaped", caption, caption)
    assert _frequent(data) == ["1girl", r"jeanne d'arc alter santa lily \(fate\)", "smile"]
    # So the preview shows the character, not just solo.
    route = registry.get("illustrious")
    prompt = train.default_sample_prompt(route, "", _frequent(data))
    assert prompt.startswith(r"1girl, jeanne d'arc alter santa lily \(fate\), smile, cowboy shot")


def test_frequent_tags_read_captions_as_the_trainer_does(tmp_path):
    # The trainer reads only the first line unless multiline captions are on.
    sentence = "A girl with long blonde hair stands by the window."
    data = _captions(tmp_path / "data", f"1girl, solo, smile\n{sentence}", f"1girl, solo, smile\n{sentence}")
    assert _frequent(data) == ["1girl", "solo", "smile"]
    assert _frequent(data, {"enable_multiline_caption": True}) == []
    # With multiline captions on, separator lines are not tags, whatever the separator and wherever it is set.
    caption = "1girl, solo\n@@@ # tags\n1girl, smile"
    data = _captions(tmp_path / "sep", caption, caption)
    on = {"enable_multiline_caption": True, "multiline_caption_separator": "@@@"}
    assert _frequent(data, on) == ["1girl", "solo", "smile"]
    config = dataset.build_dataset_config(data)
    config["datasets"][0]["subsets"][0].update(on)
    assert dataset.frequent_tags(config, {}) == ["1girl", "solo", "smile"]
    assert dataset.trained_captions(config, {}) == ["1girl, solo\n1girl, smile"] * 2


def test_build_dataset_config_structure(image_dataset):
    cfg = dataset.build_dataset_config(image_dataset, num_repeats=5, class_tokens="mychar")
    assert cfg["general"]["resolution"] == 1024
    assert cfg["general"]["enable_bucket"] is True
    [ds] = cfg["datasets"]
    assert ds["batch_size"] == 2
    [subset] = ds["subsets"]
    assert subset["image_dir"] == str(image_dataset.resolve())
    assert subset["num_repeats"] == 5
    assert subset["class_tokens"] == "mychar"


# The pinned config_util.py schema each key must be in. [general] accepts the union of these schemas.
_SCHEMA_SCOPES = {
    "SUBSET_ASCENDABLE_SCHEMA": {"num_repeats", "caption_prefix", "shuffle_caption", "keep_tokens", "flip_aug"},
    "DB_SUBSET_ASCENDABLE_SCHEMA": {"caption_extension", "class_tokens"},
    "DB_SUBSET_DISTINCT_SCHEMA": {"image_dir"},
    "DATASET_ASCENDABLE_SCHEMA": {
        "batch_size",
        "resolution",
        "enable_bucket",
        "min_bucket_reso",
        "max_bucket_reso",
        "bucket_reso_steps",
        "bucket_no_upscale",
    },
}


def test_scope_map_covers_every_emitted_key():
    mapped = set().union(*_SCHEMA_SCOPES.values())
    assert mapped == dataset.GENERAL_KEYS | dataset.DATASET_KEYS | dataset.SUBSET_KEYS


@needs_submodule
def test_schema_keys_exist_upstream_in_the_right_scope():
    """A whole-file grep misses a key that moved to a schema where it is not valid."""
    source = (SD_SCRIPTS / "library" / "config_util.py").read_text(encoding="utf-8", errors="replace")
    for schema_name, keys in _SCHEMA_SCOPES.items():
        match = re.search(rf"{schema_name} = \{{(.*?)\n    \}}", source, re.DOTALL)
        assert match, f"{schema_name} not found in pinned config_util.py"
        block = match.group(1)
        for key in sorted(keys):
            assert f'"{key}"' in block, f"{key!r} not in {schema_name} at the pin"


def test_prepare_dataset_passthrough_for_dir(image_dataset, tmp_path):
    assert dataset.prepare_dataset(image_dataset, tmp_path / "ws") == image_dataset.resolve()


def test_prepare_dataset_returns_absolute_for_relative_source(image_dataset, tmp_path, monkeypatch):
    """The trainer and the tagger run with cwd=sd-scripts, so relative paths must be resolved."""
    monkeypatch.chdir(image_dataset.parent)
    result = dataset.prepare_dataset(image_dataset.name, tmp_path / "ws")
    assert result.is_absolute()
    assert result == image_dataset.resolve()


def test_prepare_dataset_extracts_zip_and_descends(image_dataset, tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in image_dataset.iterdir():
            zf.write(file, f"wrapped/{file.name}")
    result = dataset.prepare_dataset(archive, tmp_path / "ws")
    assert result.name == "wrapped"
    assert result.is_absolute()
    assert dataset.summarize(result)["images"] == 3


def test_prepare_dataset_reextraction_is_fresh(image_dataset, tmp_path):
    """Re-running with a corrected zip must not merge with the previous extraction."""

    def make_zip(name, files):
        archive = tmp_path / name
        with zipfile.ZipFile(archive, "w") as zf:
            for f in files:
                zf.writestr(f, "x")
        return archive

    ws = tmp_path / "ws"
    first = dataset.prepare_dataset(make_zip("v1.zip", ["a.png", "stale.png"]), ws)
    assert (first / "stale.png").exists()
    second = dataset.prepare_dataset(make_zip("v2.zip", ["a.png"]), ws)
    assert not (second / "stale.png").exists()


def test_summarize_counts_caption_coverage(image_dataset):
    summary = dataset.summarize(image_dataset)
    assert summary == {"images": 3, "captioned": 2, "uncaptioned": 1, "nested_images": 0}


def test_summarize_matches_trainer_glob_not_recursive(tmp_path):
    """sd-scripts reads image_dir without its subfolders and matches only *.ext
    and *.EXT. Our count must agree, and macOS "._*" files must not add to it."""
    root = tmp_path / "data"
    sub = root / "concept"  # not a 10_name folder, so the trainer never reads it
    sub.mkdir(parents=True)
    (sub / "a.png").write_bytes(b"x")
    (sub / "b.png").write_bytes(b"x")
    (root / "._junk.png").write_bytes(b"x")
    (root / "mixed.Jpg").write_bytes(b"x")  # sd-scripts does not train this
    (root / "upper.PNG").write_bytes(b"x")  # sd-scripts trains this
    summary = dataset.summarize(root)
    assert summary["images"] == 1
    assert summary["nested_images"] == 2


def test_default_num_repeats_bounds():
    assert dataset.default_num_repeats(0) == 1
    assert dataset.default_num_repeats(10) == 20  # clamped high
    assert dataset.default_num_repeats(400) == 1
    assert dataset.default_num_repeats(100_000) == 1


def _image(folder, name, caption=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"x")
    if caption is not None:
        (folder / name).with_suffix(".txt").write_text(caption, encoding="utf-8")


def _numbered_dataset(root):
    _image(root / "10_mychar", "a.png", "1girl, smile")
    _image(root / "10_mychar", "b.png")
    _image(root / "3_red_hair", "c.png", "1girl, red hair")
    _image(root, "top.png")
    _image(root / "raw", "unused.png")  # not a numbered folder: never trains
    (root / "5_empty").mkdir()
    _image(root / ".ipynb_checkpoints", "x.png")  # tool data: skipped without a note
    return root


def test_repeat_folder_names_parse_like_the_backend():
    assert dataset.repeat_folder("10_mychar") == (10, "mychar")
    assert dataset.repeat_folder("3_red_hair") == (3, "red_hair")
    assert dataset.repeat_folder("10_") == (10, "")
    for name in ("mychar", "0_mychar", "10mychar", "x10_mychar", "-1_mychar"):
        assert dataset.repeat_folder(name) is None, name


@needs_submodule
def test_repeat_folder_rule_matches_the_pin():
    # generate_dreambooth_subsets_config_by_subdirs: the number before the
    # first "_" is the repeats, the rest is class_tokens, below 1 is skipped.
    source = (SD_SCRIPTS / "library" / "config_util.py").read_text(encoding="utf-8")
    assert 'tokens = name.split("_")' in source
    assert "n_repeats = int(tokens[0])" in source
    assert 'caption_by_folder = "_".join(tokens[1:])' in source
    assert "if num_repeats < 1:" in source


def test_numbered_folders_train_as_their_own_subsets(tmp_path):
    root = _numbered_dataset(tmp_path / "data")
    layout = dataset.find_subsets(root)
    assert [(s.image_dir.name, s.num_repeats, s.name) for s in layout.subsets] == [
        ("data", None, ""),
        ("10_mychar", 10, "mychar"),
        ("3_red_hair", 3, "red_hair"),
    ]
    assert layout.ignored == ("raw",)
    assert layout.empty == ("5_empty",)
    notes = " ".join(dataset.layout_notes(layout))
    assert "raw" in notes and "5_empty" in notes and "1 image is in the top folder" in notes
    assert ".ipynb_checkpoints" not in notes

    cfg = dataset.build_dataset_config(root, num_repeats=7)
    [ds] = cfg["datasets"]
    subsets = {Path(s["image_dir"]).name: s for s in ds["subsets"]}
    assert set(subsets) == {"data", "10_mychar", "3_red_hair"}
    assert subsets["data"]["num_repeats"] == 7 and "class_tokens" not in subsets["data"]
    assert subsets["10_mychar"]["num_repeats"] == 10 and subsets["10_mychar"]["class_tokens"] == "mychar"
    assert subsets["3_red_hair"]["class_tokens"] == "red_hair"
    # With a trigger word the folder names stay out of the captions.
    cfg = dataset.build_dataset_config(root, num_repeats=7, caption_prefix="zkx,", folder_class_tokens=False)
    assert all("class_tokens" not in s for s in cfg["datasets"][0]["subsets"])
    assert all(s["caption_prefix"] == "zkx," for s in cfg["datasets"][0]["subsets"])


def test_summary_and_tag_report_cover_every_subset(tmp_path):
    root = _numbered_dataset(tmp_path / "data")
    summary = dataset.summarize(root)
    assert summary == {"images": 4, "captioned": 2, "uncaptioned": 2, "nested_images": 2}
    counts = {s["folder"]: (s["images"], s["captioned"]) for s in dataset.subset_counts(root)}
    assert counts == {"data": (1, 0), "10_mychar": (2, 1), "3_red_hair": (1, 1)}
    report = dataset.tag_report(root)
    assert "3 distinct tags across 2 captioned images" in report
    text = dataset.describe(root)
    assert "10_mychar: 2 images, 1 captioned, 10 repeats" in text and "raw" in text


def test_plain_folder_with_a_stray_subfolder_is_one_subset(image_dataset):
    (image_dataset / "notes").mkdir()
    layout = dataset.find_subsets(image_dataset)
    assert [s.image_dir for s in layout.subsets] == [image_dataset]
    assert layout.ignored == () and dataset.layout_notes(layout) == []


def test_zip_holding_one_numbered_folder_keeps_its_number(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("10_mychar/a.png", "x")
        zf.writestr("10_mychar/a.txt", "1girl")
    result = dataset.prepare_dataset(archive, tmp_path / "ws")
    [subset] = dataset.find_subsets(result).subsets
    assert (subset.image_dir.name, subset.num_repeats) == ("10_mychar", 10)


def test_the_same_zip_keeps_the_extracted_folder_and_its_captions(tmp_path, capsys):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.png", "x")
    ws = tmp_path / "ws"
    first = dataset.prepare_dataset(archive, ws)
    (first / "a.txt").write_text("1girl, tagged since", encoding="utf-8")
    assert (ws / dataset.EXTRACT_RECORD).is_file()  # a dotfile next to the folder, not in it
    again = dataset.prepare_dataset(archive, ws)
    assert again == first
    assert (again / "a.txt").read_text(encoding="utf-8") == "1girl, tagged since"
    assert "has not changed" in capsys.readouterr().out
    # A changed zip is extracted fresh.
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.png", "x")
        zf.writestr("b.png", "y")
    fresh = dataset.prepare_dataset(archive, ws)
    assert not (fresh / "a.txt").exists() and (fresh / "b.png").exists()


def _zip_bytes(tmp_path, files):
    archive = tmp_path / "made.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in files:
            zf.writestr(name, "x")
    return archive.read_bytes()


def test_a_zip_link_downloads_into_scratch_once_and_extracts(tmp_path, monkeypatch):
    from kohya_trainer import models

    body = _zip_bytes(tmp_path, ["5_mychar/a.png"])
    fetched = []

    def fake_download(url, target):
        fetched.append(url)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        return target

    monkeypatch.setattr(models, "download_file", fake_download)
    ws = tmp_path / "ws"
    url = "https://example.com/files/set.zip?download=1"
    result = dataset.prepare_dataset(url, ws)
    assert ws in result.parents or result == (ws / "dataset").resolve()
    [subset] = dataset.find_subsets(result).subsets
    assert subset.num_repeats == 5
    (subset.image_dir / "a.txt").write_text("kept", encoding="utf-8")
    dataset.prepare_dataset(url, ws)  # same download, same zip: the caption stays
    assert (subset.image_dir / "a.txt").read_text(encoding="utf-8") == "kept"
    assert fetched == [url, url]


class _FakeHub:
    def __init__(self, files):
        self.files = files
        self.calls = []

    def __call__(self, repo_id, repo_type=None, local_dir=None, revision=None, allow_patterns=None, **kwargs):
        import fnmatch

        self.calls.append((repo_id, repo_type))
        self.options = {"revision": revision, "allow_patterns": allow_patterns}
        for name, data in self.files.items():
            if allow_patterns and not fnmatch.fnmatch(name, allow_patterns):
                continue
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(local_dir)


def test_a_hugging_face_dataset_is_fetched_once_with_its_folders(tmp_path, monkeypatch):
    import huggingface_hub

    hub = _FakeHub({"README.md": b"card", "images/10_mychar/a.png": b"x", "images/b.png": b"x"})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    monkeypatch.chdir(tmp_path)  # no local folder called someone/set
    ws = tmp_path / "ws"
    result = dataset.prepare_dataset("someone/set", ws)
    assert result.name == "images"  # past the README, into the one folder
    assert [s.num_repeats for s in dataset.find_subsets(result).subsets] == [None, 10]
    assert hub.calls == [("someone/set", "dataset")]
    again = dataset.prepare_dataset("https://huggingface.co/datasets/someone/set", ws)
    assert again == result and len(hub.calls) == 1  # the earlier copy is used


def test_a_hub_tree_link_fetches_that_folder_at_that_revision(tmp_path, monkeypatch):
    import huggingface_hub

    files = {"README.md": b"card", "v2/10_mychar/a.png": b"x", "v2/10_mychar/a.txt": b"1girl", "old/b.png": b"x"}
    hub = _FakeHub(files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    ws = tmp_path / "ws"
    result = dataset.prepare_dataset("https://huggingface.co/datasets/someone/set/tree/v2.1/v2/10_mychar", ws)
    assert hub.options == {"revision": "v2.1", "allow_patterns": "v2/10_mychar/*"}
    [subset] = dataset.find_subsets(result).subsets  # the folder keeps its number
    assert (subset.image_dir.name, subset.num_repeats) == ("10_mychar", 10)
    assert not any(p.name == "b.png" for p in ws.rglob("*"))
    # A folder without a number is the dataset itself.
    result = dataset.prepare_dataset("https://huggingface.co/datasets/someone/set/tree/main/old", ws)
    assert result.name == "old" and hub.options["revision"] == "main"
    # The same link again uses the copy fetched before.
    dataset.prepare_dataset("https://huggingface.co/datasets/someone/set/tree/main/old", ws)
    assert len(hub.calls) == 2


def test_a_hub_dataset_of_parquet_files_is_refused_with_the_reason(tmp_path, monkeypatch):
    import huggingface_hub

    for files in ({"data/train-0000.parquet": b"x", "README.md": b"card"}, {"metadata.jsonl": b"{}"}):
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _FakeHub(files))
        with pytest.raises(ValueError, match="format is not supported"):
            dataset.prepare_dataset("https://huggingface.co/datasets/someone/parquet", tmp_path / "ws")
        for record in (tmp_path / "ws" / "hub").glob(".*.json"):
            record.unlink()  # fetch again for the next format


def test_a_bad_archive_keeps_the_folder_extracted_before(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.png", "x")
    ws = tmp_path / "ws"
    first = dataset.prepare_dataset(archive, ws)
    (first / "a.txt").write_text("kept", encoding="utf-8")
    archive.write_bytes(b"<html>not a zip</html>")  # a Drive zip still syncing, say
    with pytest.raises(ValueError, match="not a zip file that opens"):
        dataset.prepare_dataset(archive, ws)
    assert (first / "a.txt").read_text(encoding="utf-8") == "kept"


def test_a_damaged_file_inside_the_zip_keeps_the_folder_extracted_before(tmp_path):
    """The zip opens, but one file in it does not read back."""
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("a.png", "xxxxxxxx")
    ws = tmp_path / "ws"
    first = dataset.prepare_dataset(archive, ws)
    (first / "a.txt").write_text("kept", encoding="utf-8")
    raw = bytearray(archive.read_bytes())
    raw[30 + len("a.png")] ^= 0xFF  # the first byte of a.png's data, after its local header
    archive.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="damaged and does not extract"):
        dataset.prepare_dataset(archive, ws)
    assert (first / "a.txt").read_text(encoding="utf-8") == "kept"
    assert (first / "a.png").read_bytes() == b"xxxxxxxx"  # not overwritten by the damaged copy
    assert not any(p.name.endswith(".partial") for p in first.parent.iterdir())


def test_a_hub_dataset_with_metadata_captions_gets_a_note(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _FakeHub({"a.png": b"x", "metadata.jsonl": b"{}"}))
    dataset.prepare_dataset("someone/imagefolder", tmp_path / "ws")
    assert "metadata.jsonl, which the trainer does not read" in capsys.readouterr().out


def test_a_link_that_gives_a_web_page_is_deleted_and_named(tmp_path, monkeypatch):
    from kohya_trainer import models

    saved = []

    def fake_download(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            target.write_bytes(b"<html>sign in</html>")
        saved.append(target)
        return target

    monkeypatch.setattr(models, "download_file", fake_download)
    url = "https://www.dropbox.com/s/x/data.zip?dl=0"
    with pytest.raises(ValueError, match="did not give a working .zip file") as error:
        dataset.prepare_dataset(url, tmp_path / "ws")
    assert url in str(error.value)
    assert not saved[0].exists()  # the next run downloads it again


def test_a_large_folder_number_gets_a_note(tmp_path):
    (tmp_path / "2024_trip").mkdir()
    (tmp_path / "2024_trip" / "a.png").write_bytes(b"x")
    (tmp_path / "0_extra").mkdir()
    (tmp_path / "0_extra" / "b.png").write_bytes(b"x")
    notes = dataset.layout_notes(dataset.find_subsets(tmp_path))
    assert any("2024_trip trains 2024 times per epoch" in note and "10_trip" in note for note in notes)
    assert any("a number of 1 or more" in note and "0_extra" in note for note in notes)


def test_a_hugging_face_dataset_holding_one_zip_is_extracted(tmp_path, monkeypatch):
    import huggingface_hub

    hub = _FakeHub({"data.zip": _zip_bytes(tmp_path, ["a.png", "a.txt"]), ".gitattributes": b""})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    result = dataset.prepare_dataset("https://huggingface.co/datasets/someone/zipped", tmp_path / "ws")
    assert result == (tmp_path / "ws" / "dataset").resolve()
    assert dataset.summarize(result)["captioned"] == 1


def test_a_local_folder_wins_over_a_hub_name(tmp_path, monkeypatch):
    import huggingface_hub

    def no_hub(*args, **kwargs):
        raise AssertionError("fetched from the Hub although the folder exists")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_hub)
    (tmp_path / "user" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert dataset.prepare_dataset("user/name", tmp_path / "ws") == (tmp_path / "user" / "name").resolve()


def test_unknown_sources_name_the_accepted_forms(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import RepositoryNotFoundError

    def missing(*args, **kwargs):
        # Built without an HTTP response, which only the real client has.
        error = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        Exception.__init__(error, "404")
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Hugging Face dataset") as error:
        dataset.prepare_dataset("nobody/nothing", tmp_path / "ws")
    assert "HF_TOKEN" in str(error.value)
    with pytest.raises(FileNotFoundError, match=r"an https link to a \.zip"):
        dataset.prepare_dataset(tmp_path / "missing.zip", tmp_path / "ws")
    with pytest.raises(ValueError, match="Google Drive share link"):
        dataset.prepare_dataset("https://drive.google.com/file/d/abc/view?usp=sharing", tmp_path / "ws")
    with pytest.raises(ValueError, match="not a link to a .zip"):
        dataset.prepare_dataset("https://example.com/gallery", tmp_path / "ws")


def test_a_full_disk_while_extracting_keeps_the_zip_and_the_old_folder(tmp_path, monkeypatch):
    """A disk error is not the zip's fault, so it shows as it is."""
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.png", "x")
    ws = tmp_path / "ws"
    first = dataset.prepare_dataset(archive, ws)
    (first / "a.txt").write_text("kept", encoding="utf-8")
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("b.png", "y")  # a changed zip, so it is extracted again

    def no_space(self, path=None, members=None, pwd=None):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", no_space)
    with pytest.raises(OSError, match="No space left") as error:
        dataset.prepare_dataset(archive, ws)
    assert not isinstance(error.value, ValueError)
    assert archive.is_file()
    assert (first / "a.txt").read_text(encoding="utf-8") == "kept"
    assert not any(p.name.endswith(".partial") for p in first.parent.iterdir())


@pytest.mark.parametrize(
    "caption, expected",
    [
        ("1girl, solo", 1),
        ("a long sentence.\nIt goes on here.", 1),  # line breaks alone make one candidate
        ("tags, here\n---===---\na sentence about the same image.", 2),
        ("first\n  ---===---  # a comment\nsecond\n---===---\nthird", 3),
        ("only one\n---===---\n\n", 1),  # an empty candidate is dropped
        ("---===---\nonly one", 1),
    ],
)
def test_candidate_count_splits_like_the_backend(caption, expected):
    assert dataset._candidate_count(caption, dataset.DEFAULT_MULTILINE_CAPTION_SEPARATOR) == expected


def test_captions_with_candidates_needs_multiline_and_reads_the_separator(image_dataset):
    (image_dataset / "a.txt").write_text("tags\n---===---\nsentence", encoding="utf-8")
    (image_dataset / "b.txt").write_text("tags\n###\nsentence", encoding="utf-8")
    config = dataset.build_dataset_config(image_dataset)
    assert dataset.captions_with_candidates(config, {}) == []
    found = dataset.captions_with_candidates(config, {"enable_multiline_caption": True})
    assert [image.name for image in found] == ["a.png"]
    found = dataset.captions_with_candidates(
        config, {"enable_multiline_caption": True, "multiline_caption_separator": "###"}
    )
    assert [image.name for image in found] == ["b.png"]

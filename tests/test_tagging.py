from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from kohya_trainer import environment, tagging

from conftest import SD_SCRIPTS, needs_submodule


def _fake_backend(root, body: str):
    """A backend folder whose tagger script is a tiny stand-in."""
    script = root / "finetune" / "tag_images_by_wd14_tagger.py"
    script.parent.mkdir(parents=True)
    script.write_text(body, encoding="utf-8", newline="\n")
    return root


@pytest.fixture(autouse=True)
def _no_installs(monkeypatch):
    monkeypatch.setattr(tagging.environment, "ensure_tagger_deps", lambda: None)


def test_tagger_failure_shows_output_and_says_tagging_failed(tmp_path, capsys):
    backend = _fake_backend(
        tmp_path / "bad",
        "print('loading tagger model')\nprint('onnxruntime not found')\nraise SystemExit(2)\n",
    )
    with pytest.raises(RuntimeError) as error:
        tagging.tag_images(tmp_path, sd_scripts_dir=backend)
    out = capsys.readouterr().out
    assert "loading tagger model" in out and "onnxruntime not found" in out
    message = str(error.value)
    assert message.startswith("tagging failed")
    assert "training" not in message
    assert "code 2" in message
    assert "onnxruntime not found" in message


def test_tagger_success_shows_output(tmp_path, capsys):
    backend = _fake_backend(tmp_path / "good", "print('tagged 3 images')\n")
    tagging.tag_images(tmp_path, sd_scripts_dir=backend)
    assert "tagged 3 images" in capsys.readouterr().out


def test_streamed_output_keeps_characters_split_across_reads(tmp_path, capsys):
    # One write longer than a read, of 3-byte characters, splits some of them.
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\nsys.stdout.buffer.write(('\\u3042' * 3000).encode('utf-8'))\n",
        encoding="utf-8",
        newline="\n",
    )
    code, tail = environment.run_streamed([sys.executable, str(child)], cwd=tmp_path)
    assert code == 0
    assert capsys.readouterr().out == "あ" * 3000
    assert "�" not in tail


def test_interrupt_stops_the_child(tmp_path, monkeypatch):
    # The child writes a marker only if it is still running 1.5 seconds in.
    marker = tmp_path / "still-running"
    child = tmp_path / "child.py"
    child.write_text(
        f"import time\nprint('started', flush=True)\ntime.sleep(1.5)\nopen({str(marker)!r}, 'w').close()\n",
        encoding="utf-8",
        newline="\n",
    )

    class Interrupted:
        """A cell output that raises like Colab's stop button does."""

        def write(self, text):
            raise KeyboardInterrupt

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", Interrupted())
    with pytest.raises(KeyboardInterrupt):
        environment.run_streamed([sys.executable, str(child)], cwd=tmp_path)
    monkeypatch.undo()
    time.sleep(2.5)
    assert not marker.exists(), "the child kept running after the interrupt"


def _flags(**options) -> list[str]:
    return tagging.build_tag_command("d", sd_scripts_dir="sd", **options)[3:]


def test_eva02_large_is_the_default_and_swinv2_is_gone():
    assert tagging.DEFAULT_TAGGER == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert tagging.TAGGERS == ("SmilingWolf/wd-eva02-large-tagger-v3", "SmilingWolf/wd-vit-large-tagger-v3")
    assert f"--repo_id={tagging.DEFAULT_TAGGER}" in _flags()


def test_undesired_tags_match_with_or_without_underscores():
    flags = _flags(undesired_tags="long_hair, simple background, ^_^ ,")
    assert "--undesired_tags=long hair,simple background,^_^" in flags
    # Without remove_underscore the tagger keeps underscores, so the list does too.
    flags = _flags(undesired_tags="long_hair", remove_underscore=False)
    assert "--undesired_tags=long_hair" in flags
    assert not any(flag.startswith("--undesired_tags") for flag in _flags(undesired_tags=" , "))


def test_tag_order_and_rating_flags_are_opt_in():
    assert "--character_tags_first" not in _flags()
    assert "--use_rating_tags" not in _flags()
    flags = _flags(character_tags_first=True, use_rating_tags=True)
    assert "--character_tags_first" in flags and "--use_rating_tags" in flags


@needs_submodule
def test_tagger_flags_and_underscore_rule_match_the_pin():
    source = (SD_SCRIPTS / "finetune" / "tag_images_by_wd14_tagger.py").read_text(encoding="utf-8")
    for flag in ("--character_tags_first", "--use_rating_tags", "--remove_underscore", "--undesired_tags"):
        assert f'"{flag}"' in source, flag
    assert 'tag.replace("_", " ") if len(tag) > 3 else tag' in source
    assert tagging.normalize_tag("long_hair") == "long hair"
    assert tagging.normalize_tag("^_^") == "^_^"


def test_each_subset_folder_is_tagged_and_ignored_folders_are_not(tmp_path):
    record = tmp_path / "tagged.txt"
    backend = _fake_backend(
        tmp_path / "backend",
        f"import sys\nopen({str(record)!r}, 'a', encoding='utf-8').write(sys.argv[1] + chr(10))\n",
    )
    data = tmp_path / "data"
    for folder in (data / "10_mychar", data / "3_red", data / "raw", data):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.png").write_bytes(b"x")
    (data / "5_empty").mkdir()
    tagging.tag_images(data, sd_scripts_dir=backend)
    tagged = sorted(Path(line).name for line in record.read_text(encoding="utf-8").splitlines())
    assert tagged == ["10_mychar", "3_red", "data"]


def test_captions_are_backed_up_before_the_tagger_replaces_them(tmp_path, capsys):
    backend = _fake_backend(tmp_path / "backend", "pass\n")
    data = tmp_path / "data"
    (data / "10_mychar").mkdir(parents=True)
    for folder in (data, data / "10_mychar"):
        (folder / "a.png").write_bytes(b"x")
        (folder / "a.txt").write_text(f"hand written in {folder.name}", encoding="utf-8")
    (data / "b.png").write_bytes(b"x")  # no caption: nothing to keep
    backups = tmp_path / "ws" / "captions-backup"
    tagging.tag_images(data, sd_scripts_dir=backend, backup_dir=backups, append_tags=True)
    assert not backups.exists()  # appending keeps the captions, so no copy is needed
    tagging.tag_images(data, sd_scripts_dir=backend, backup_dir=backups)
    [stamp] = list(backups.iterdir())
    assert (stamp / "a.txt").read_text(encoding="utf-8") == "hand written in data"
    assert (stamp / "10_mychar" / "a.txt").read_text(encoding="utf-8") == "hand written in 10_mychar"
    assert sorted(p.name for p in stamp.rglob("*") if p.is_file()) == ["a.txt", "a.txt"]
    assert str(stamp) in capsys.readouterr().out


def test_remove_tags_edits_every_caption_with_or_without_underscores(tmp_path, capsys):
    data = tmp_path / "data"
    (data / "3_red").mkdir(parents=True)
    for folder, caption in ((data, "1girl, long_hair, smile, ^_^"), (data / "3_red", "1girl, long hair")):
        (folder / "a.png").write_bytes(b"x")
        (folder / "a.txt").write_text(caption, encoding="utf-8")
    (data / "stray.txt").write_text("long hair", encoding="utf-8")  # no image: not a caption
    changed = tagging.remove_tags(data, "long hair, ^_^, not_there")
    assert changed == 2
    assert (data / "a.txt").read_text(encoding="utf-8") == "1girl, smile"
    assert (data / "3_red" / "a.txt").read_text(encoding="utf-8") == "1girl"
    assert (data / "stray.txt").read_text(encoding="utf-8") == "long hair"
    assert "removed 3 tags from 2 caption files" in capsys.readouterr().out
    assert tagging.remove_tags(data, "long_hair") == 0
    assert "none of these tags" in capsys.readouterr().out


def test_prune_into_trigger_takes_out_only_common_hair_and_eye_tags(tmp_path, capsys):
    data = tmp_path / "data"
    (data / "10_mychar").mkdir(parents=True)
    captions = {
        data / "a.txt": "1girl, long_hair, blonde_hair, green eyes, school uniform, closed eyes",
        data / "b.txt": "1girl, long hair, blonde hair, green_eyes, school uniform, hair over one eye",
        data / "10_mychar" / "c.txt": "1girl, long hair, blonde hair, red eyes, sundress, floating hair",
        data / "10_mychar" / "d.txt": "1girl, long hair, short hair, sundress, closed eyes, hair over one eye",
    }
    for caption, text in captions.items():
        caption.with_suffix(".png").write_bytes(b"x")
        caption.write_text(text, encoding="utf-8")
    pruned = tagging.prune_into_trigger(data)
    # In at least half of the four captions, with or without underscores.
    # One-image tags, outfits and pose tags such as closed eyes stay.
    assert pruned == ["long hair", "blonde hair", "green eyes"]
    assert (data / "a.txt").read_text(encoding="utf-8") == "1girl, school uniform, closed eyes"
    assert (data / "10_mychar" / "c.txt").read_text(encoding="utf-8") == "1girl, red eyes, sundress, floating hair"
    assert (data / "10_mychar" / "d.txt").read_text(encoding="utf-8") == (
        "1girl, short hair, sundress, closed eyes, hair over one eye"
    )
    out = capsys.readouterr().out
    assert "long hair (4 of 4), blonde hair (3 of 4), green eyes (2 of 4)" in out
    assert "the trigger word learns them" in out
    assert tagging.prune_into_trigger(data) == []
    assert "none were taken out" in capsys.readouterr().out
    # A lower share takes out tags in fewer captions.
    assert tagging.prune_into_trigger(data, min_share=0.25) == ["red eyes", "short hair"]


def test_appearance_tags_name_who_it_is_not_how_they_pose():
    for tag in ("blonde hair", "light blue hair", "platinum blonde hair", "very long hair", "heterochromia"):
        assert tag in tagging.APPEARANCE_TAGS, tag
    for tag in ("aqua eyes", "two-tone hair", "multicolored hair", "gradient hair", "streaked hair"):
        assert tag in tagging.APPEARANCE_TAGS, tag
    for tag in ("very short hair", "colored inner hair", "split-color hair", "multicolored eyes"):
        assert tag in tagging.APPEARANCE_TAGS, tag
    for tag in ("closed eyes", "hair over one eye", "floating hair", "hair ornament", "school uniform"):
        assert tag not in tagging.APPEARANCE_TAGS, tag

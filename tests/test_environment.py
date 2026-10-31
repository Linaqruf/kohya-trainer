"""environment.py: everything that depends on Colab or the machine."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import types
from pathlib import Path

import pytest

from kohya_trainer import environment

from conftest import SD_SCRIPTS, needs_submodule


@needs_submodule
def test_sd_scripts_dir_env_override(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere"
    override.mkdir()
    (override / "train_network.py").write_text("# marker", encoding="utf-8")
    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(override))
    assert environment.sd_scripts_dir() == override


@needs_submodule
def test_sd_scripts_dir_bad_override_falls_back_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(tmp_path / "nope"))
    assert environment.sd_scripts_dir() == SD_SCRIPTS


def test_sd_scripts_dir_missing_raises_with_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHYA_TRAINER_SD_SCRIPTS", str(tmp_path / "nope"))
    monkeypatch.setattr(environment, "repo_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="recurse-submodules"):
        environment.sd_scripts_dir()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.10.0+cu126", (2, 10)),
        ("2.6.0", (2, 6)),
        ("2.11.0.dev20260101+cu130", (2, 11)),
    ],
)
def test_torch_version_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: raw)
    assert environment.torch_version() == expected


def test_torch_version_missing(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert environment.torch_version() is None


class FakeSite:
    """Installed torch and torchvision that pip calls can change.

    ``torchvision`` is the torch pin in torchvision's metadata, or None when it is not installed.
    ``requirements_torch`` is the torch left after the backend requirements install, or None when removed.
    """

    KEEP = object()

    def __init__(self, monkeypatch, torch, torchvision="torch==2.8.0", requirements_torch=KEEP):
        self.torch = torch
        self.torchvision = torchvision
        self.requirements_torch = requirements_torch
        self.calls: list[tuple[str, ...]] = []
        monkeypatch.setattr(importlib.metadata, "version", self.version)
        monkeypatch.setattr(importlib.metadata, "requires", self.requires)
        monkeypatch.setattr(environment, "_pip", self.pip)
        monkeypatch.setattr(environment, "ensure_accelerate_config", lambda: None)
        monkeypatch.setattr(environment, "sd_scripts_dir", lambda: Path("/fake/sd-scripts"))
        monkeypatch.setattr(environment, "gpu_name", lambda: None)

    def version(self, name):
        if name != "torch" or self.torch is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return self.torch

    def requires(self, name):
        if name != "torchvision" or self.torchvision is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return ["numpy", self.torchvision, "pillow!=8.3.*,>=5.3.0"]

    def pip(self, *args, **kwargs):
        self.calls.append(args)
        if "--index-url" in args:
            # The strict install brings a matching pair.
            self.torch, self.torchvision = "2.8.0+cu128", "torch==2.8.0"
        elif args[:2] == ("-r", "requirements.txt") and self.requirements_torch is not self.KEEP:
            self.torch = self.requirements_torch


def test_install_keeps_a_new_enough_torch_and_replaces_an_old_one(monkeypatch):
    site = FakeSite(monkeypatch, torch="2.8.0+cu126")
    assert environment.install()["torch"] == "2.8"
    assert [call[:2] for call in site.calls] == [("-r", "requirements.txt")], "new torch must be kept"

    site = FakeSite(monkeypatch, torch="2.4.1+cu121", torchvision="torch==2.4.1")
    assert environment.install()["torch"] == "2.8"
    assert any("--index-url" in call for call in site.calls), "old torch must trigger the strict install"


def test_install_raises_when_requirements_replace_the_torch(monkeypatch):
    FakeSite(monkeypatch, torch="2.8.0+cu126", requirements_torch="2.9.0")
    with pytest.raises(RuntimeError) as err:
        environment.install()
    assert "torch 2.9.0 is installed" in str(err.value) and "built for torch 2.8.0" in str(err.value)


def test_install_still_raises_when_setup_runs_again(monkeypatch):
    # After a swap, a second run sees the replaced torch as the starting point.
    FakeSite(monkeypatch, torch="2.8.0+cu126", requirements_torch="2.9.0")
    with pytest.raises(RuntimeError):
        environment.install()
    with pytest.raises(RuntimeError):
        environment.install()


def test_install_raises_when_requirements_remove_torch(monkeypatch):
    FakeSite(monkeypatch, torch="2.8.0+cu126", requirements_torch=None)
    with pytest.raises(RuntimeError, match="2.8.0"):
        environment.install()


def test_install_raises_when_the_strict_pair_is_replaced(monkeypatch):
    # Start from a pair only the strict install turns into 2.8.0.
    site = FakeSite(monkeypatch, torch="2.7.0+cu126", torchvision="torch==2.7.0", requirements_torch="2.9.0")
    with pytest.raises(RuntimeError, match="built for torch 2.8.0"):
        environment.install(strict_torch=True)
    assert any("--index-url" in call for call in site.calls)


@pytest.mark.parametrize(
    "torch, pin, ok",
    [
        ("2.8.0+cu126", "torch==2.8.0", True),  # a pin without a local tag accepts any build
        ("2.8.0", "torch==2.8.0", True),
        ("2.8.0+cu126", "torch (==2.8.0)", True),  # older metadata style
        ("2.8.0+cu126", "torch==2.8.0+cu126", True),
        ("2.8.0+cu126", "torch==2.8.0+cpu", False),  # a pin with a local tag must match exactly
        ("2.9.0+cu126", "torch==2.8.0", False),
        ("2.9.0", "torch>=2.0", True),  # no exact pin, nothing to compare
        ("2.9.0", None, True),  # no torchvision installed
    ],
)
def test_torch_pair_mismatch(monkeypatch, torch, pin, ok):
    FakeSite(monkeypatch, torch=torch, torchvision=pin)
    assert (environment.torch_pair_mismatch() is None) is ok


def test_get_secret_env_fallback_and_load_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "unit-test-value")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    assert environment.get_secret("HF_TOKEN") == "unit-test-value"
    assert environment.get_secret("WANDB_API_KEY") is None
    assert environment.load_tokens() == ["HF_TOKEN"]


class NotebookAccessError(Exception):
    """Named as in google.colab.userdata: the secret exists, but this notebook has no access to it."""


class SecretNotFoundError(Exception):
    """Named as in google.colab.userdata."""


class TimeoutException(Exception):
    """Named as in google.colab.userdata: no answer to Colab's access dialog in time."""


@pytest.mark.parametrize(
    ("error", "noted"), [(NotebookAccessError, True), (TimeoutException, True), (SecretNotFoundError, False)]
)
def test_get_secret_names_a_secret_the_notebook_may_not_read(monkeypatch, capsys, error, noted):
    def get(key):
        raise error(key)

    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=get)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(environment, "is_colab", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "unit-test-value")
    assert environment.get_secret("HF_TOKEN") == "unit-test-value"
    out = capsys.readouterr().out
    assert ("turn on Notebook access for HF_TOKEN" in out) is noted
    assert ("Grant access when Colab asks" in out) is noted
    assert ("run this cell again" in out) is noted
    assert "unit-test-value" not in out


@pytest.mark.parametrize(
    "broken",
    [
        "unit-test-value\r\nunit-test-value",  # the token pasted twice, one per line
        "unit-test-value unit-test-value",
        "\ufeffunit-test-value",  # a byte-order mark, which strip() keeps
        "unit-test\u200bvalue",  # a zero-width space from a web page
    ],
)
def test_load_tokens_refuses_a_broken_value(monkeypatch, capsys, broken):
    """A broken value fails every request, and the HTTP library prints the header, token included.
    huggingface_hub reads Colab's HF_TOKEN secret itself, so it must not use it either."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr(environment, "_hf_token_blocked", False)
    # setenv, not delenv: then the value load_tokens writes is undone afterwards.
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", broken)
    monkeypatch.setenv("WANDB_API_KEY", "other-value")
    assert environment.load_tokens() == ["WANDB_API_KEY"]
    assert "HF_TOKEN" not in environment.os.environ
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is True
    assert environment.os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    out = capsys.readouterr().out
    assert "HF_TOKEN secret has a space, line break or hidden character inside" in out
    assert "Restart session" in out
    assert "unit-test" not in out, "the value must never be printed"


def test_a_fixed_hf_token_is_sent_again(monkeypatch):
    """After a refusal and a fix, huggingface_hub's own token lookup must be back on."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr(environment, "_hf_token_blocked", False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "unit-test-value unit-test-value")
    assert environment.load_tokens() == []
    monkeypatch.setenv("HF_TOKEN", "unit-test-value")
    assert environment.load_tokens() == ["HF_TOKEN"]
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is False
    assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in environment.os.environ
    assert environment._hf_token_blocked is False


def test_a_switch_the_person_set_stays_theirs(monkeypatch):
    """Implicit tokens turned off on purpose stay off, through a refusal and a fix."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(environment, "_hf_token_blocked", False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "unit-test-value unit-test-value")
    assert environment.load_tokens() == []
    monkeypatch.setenv("HF_TOKEN", "unit-test-value")
    assert environment.load_tokens() == ["HF_TOKEN"]
    assert constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is True
    assert environment.os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_load_tokens_trims_a_line_break(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "unit-test-value\n")
    assert environment.load_tokens() == ["HF_TOKEN"]
    assert environment.os.environ["HF_TOKEN"] == "unit-test-value"


def _fake_drive(monkeypatch, errors):
    """A google.colab.drive whose mount raises the given errors in turn."""
    calls = []

    class FakeDrive:
        @staticmethod
        def mount(mountpoint, timeout_ms=None):
            calls.append(timeout_ms)
            if errors:
                raise errors.pop(0)

    colab = types.ModuleType("google.colab")
    colab.drive = FakeDrive
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(environment, "is_colab", lambda: True)
    return calls


def test_mount_drive_tries_twice(monkeypatch):
    # The first mount in a new runtime can fail while access is granted. A second try mounts.
    calls = _fake_drive(monkeypatch, [ValueError("mount failed")])
    assert environment.mount_drive("/content/drive") == Path("/content/drive")
    assert calls == [300_000, 300_000]


def test_mount_drive_gives_up_after_the_second_try(monkeypatch):
    calls = _fake_drive(monkeypatch, [ValueError("mount failed"), ValueError("mount failed")])
    with pytest.raises(ValueError, match="mount failed"):
        environment.mount_drive("/content/drive")
    assert len(calls) == 2


def test_mount_drive_does_not_retry_other_errors(monkeypatch):
    calls = _fake_drive(monkeypatch, [ValueError("Mountpoint must not already contain files")])
    with pytest.raises(ValueError, match="must not already contain"):
        environment.mount_drive("/content/drive")
    assert len(calls) == 1


def test_wandb_login_available(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(environment.Path, "home", lambda: tmp_path)
    assert environment.wandb_login_available() is False
    # An unrelated netrc (for example a git credential) is not a wandb login.
    (tmp_path / ".netrc").write_text("machine github.com login x", encoding="utf-8")
    assert environment.wandb_login_available() is False
    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password k", encoding="utf-8")
    assert environment.wandb_login_available() is True
    (tmp_path / ".netrc").unlink()
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert environment.wandb_login_available() is True
    monkeypatch.delenv("WANDB_API_KEY")
    # Offline mode needs no credentials, so it must not be refused.
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert environment.wandb_login_available() is True


def test_ensure_wandb_installs_only_when_missing(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(environment, "_pip", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: None)
    environment.ensure_wandb()
    assert calls == [("wandb",)]
    calls.clear()
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: object())
    environment.ensure_wandb()
    assert calls == []


def test_default_workspace_and_scratch_off_colab():
    assert not environment.is_colab()
    assert environment.default_workspace() == Path("runs").resolve()
    scratch = environment.scratch_dir("proj")
    assert scratch.name == "proj"
    assert ".scratch" in scratch.parts


# --- streamed backend output -------------------------------------------------

LOKR = "WARNING  LoKr: lora_dim 10000 is large for dim=640 and factor=8, using full matrix mode."


def test_repeat_filter_shows_the_first_line_and_counts_the_rest():
    quiet = environment._LineFilter()
    out = quiet.feed("loading\n" + f"{LOKR}\n" * 5 + "network created\n")
    assert out == f"loading\n{LOKR}\n(4 more lines like the one above were hidden)\nnetwork created\n"
    # Later repeats stay hidden, and the count is given at the end of the run.
    assert quiet.feed(f"{LOKR}\n{LOKR}\n") == ""
    assert quiet.finish() == "(2 more lines like the one above were hidden)\n"
    assert quiet.finish() == ""


def test_repeat_filter_passes_progress_bars_as_they_arrive():
    quiet = environment._LineFilter()
    assert quiet.feed("\r  10%|#") == "\r  10%|#"
    assert quiet.feed("#   |") == "#   |"
    assert quiet.feed("\r  20%|##") == "\r  20%|##"
    assert quiet.feed("\n") == "\n"
    # An ordinary line is held only until its line break.
    assert quiet.feed("epoch 1/") == ""
    assert quiet.feed("10\nsteps") == "epoch 1/10\n"
    assert quiet.finish() == "steps"


def test_repeat_filter_keeps_a_line_split_across_reads():
    quiet = environment._LineFilter()
    first, second = LOKR[:20], LOKR[20:]
    assert quiet.feed(first) == ""
    assert quiet.feed(second + "\n") == LOKR + "\n"
    assert quiet.feed(first) + quiet.feed(second + "\nnext\n") == "(1 more line like the one above was hidden)\nnext\n"


def test_repeat_filter_hides_the_wrapped_rest_of_a_hidden_message():
    quiet = environment._LineFilter()
    message = "WARNING  LoKr: lora_dim 10000 is large for\n                  dim=640 and factor=8\n"
    out = quiet.feed(message * 3 + "other output\n  an indented line\n")
    assert out == message + "(2 more lines like the one above were hidden)\nother output\n  an indented line\n"


def test_repeat_filter_shows_the_next_record_when_rich_blanks_its_time():
    # rich wraps a long message at the message column and blanks the time column for records
    # in the same second, so the next record starts with fewer spaces.
    first = (
        "[10/27/26 13:12:01] WARNING  LoKr: lora_dim 10000 is large for     lokr.py:175\n"
        "                             dim=640 and factor=8, using full\n"
        "                             matrix mode.\n"
    )
    repeat = (
        "                    WARNING  LoKr: lora_dim 10000 is large for     lokr.py:175\n"
        "                             dim=1280 and factor=8, using full\n"
        "                             matrix mode.\n"
    )
    record = "                    INFO     create LoKr for UNet/DiT: 722 modules.   network_base.py:289\n"
    quiet = environment._LineFilter()
    out = quiet.feed(first + repeat * 2 + record + "prepare optimizer, data loader etc.\n")
    assert out == first + "(2 more lines like the one above were hidden)\n" + record + (
        "prepare optimizer, data loader etc.\n"
    )


def test_line_filter_hides_hugging_face_requests_with_their_wrapped_rest():
    request = (
        "2026-10-30 10:50:14 INFO     HTTP Request: GET                    _client.py:1025\n"
        "                             https://huggingface.co/api/models/o\n"
        "                             penai/clip-vit-large-patch14/tree/m\n"
        '                             ain "HTTP/1.1 200 OK"\n'
    )
    record = "                    INFO     prepare optimizer, data loader etc.  train_network.py:500\n"
    quiet = environment._LineFilter()
    out = quiet.feed("loading tokenizer\n" + request * 3 + record)
    # Hidden every time, with no count.
    assert out == "loading tokenizer\n" + record
    assert quiet.finish() == ""


def test_line_filter_hides_the_tagger_cpu_note_and_keeps_other_onnxruntime_warnings():
    fallback = (
        "2026-10-30 10:49:13.100 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:965 "
        "CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider.\n"
    )
    note = (
        "2026-10-30 10:49:14.887 [W:onnxruntime:, session_state.cc:1598 VerifyEachNodeIsAssignedToAnEp] "
        "Some nodes were not assigned to the preferred execution providers.\n"
        "2026-10-30 10:49:14.887 [W:onnxruntime:, session_state.cc:1600 VerifyEachNodeIsAssignedToAnEp] "
        "Rerunning with verbose output on a non-minimal build will show node assignments.\n"
    )
    quiet = environment._LineFilter()
    assert quiet.feed(fallback + note + "\r100%|#####| 5/5\n") == fallback + "\r100%|#####| 5/5\n"


def test_line_filter_counts_only_the_repeats():
    request = 'INFO     HTTP Request: HEAD https://huggingface.co/x "HTTP/1.1 200 OK"\n'
    quiet = environment._LineFilter()
    out = quiet.feed(f"{LOKR}\n{request}{LOKR}\n{request}done\n")
    assert out == f"{LOKR}\n(1 more line like the one above was hidden)\ndone\n"


def test_repeat_filter_reads_windows_line_breaks_split_across_reads():
    quiet = environment._LineFilter()
    assert quiet.feed(f"{LOKR}\r") == ""
    assert quiet.feed(f"\n{LOKR}\r\nnext\r") == f"{LOKR}\r\n"
    assert quiet.feed("\n") == "(1 more line like the one above was hidden)\nnext\r\n"
    assert quiet.feed("\r 5%|") == "\r 5%|"
    assert quiet.finish() == ""


def _child(tmp_path, body: str):
    child = tmp_path / "child.py"
    child.write_text(body, encoding="utf-8", newline="\n")
    return child


def test_streamed_run_quiets_a_flood_and_keeps_progress_and_lines(tmp_path, capsys):
    child = _child(
        tmp_path,
        "import sys, time\n"
        "print('start', flush=True)\n"
        f"for _ in range(722):\n    print({LOKR!r})\n"
        "sys.stdout.flush()\n"
        "for i in range(3):\n"
        "    sys.stdout.write(f'\\r{i * 50}%|bar|')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.05)\n"
        "print()\n"
        "print('done', flush=True)\n"
        "raise SystemExit(3)\n",
    )
    code, tail = environment.run_streamed([sys.executable, str(child)], cwd=tmp_path)
    out = capsys.readouterr().out.replace("\r\n", "\n")  # a Windows child ends lines with \r\n
    assert code == 3
    assert out == (
        f"start\n{LOKR}\n(721 more lines like the one above were hidden)\n\r0%|bar|\r50%|bar|\r100%|bar|\ndone\n"
    )
    assert tail.endswith("done")


def test_child_env_adds_the_syntax_warning_filter(monkeypatch):
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)
    assert environment._child_env()["PYTHONWARNINGS"] == "ignore::SyntaxWarning"
    # Ours goes first: Python uses the last filter that matches, so the user's filters win.
    monkeypatch.setenv("PYTHONWARNINGS", "error::SyntaxWarning")
    assert environment._child_env()["PYTHONWARNINGS"] == "ignore::SyntaxWarning,error::SyntaxWarning"


def test_child_env_puts_every_tqdm_bar_on_one_row(monkeypatch):
    monkeypatch.delenv("TQDM_POSITION", raising=False)
    assert environment._child_env()["TQDM_POSITION"] == "0"
    monkeypatch.setenv("TQDM_POSITION", "2")
    assert environment._child_env()["TQDM_POSITION"] == "2"


def _rows(out: str) -> list[str]:
    """The rows a notebook cell shows.

    \\r writes over the row from its start and keeps any longer tail, as
    Jupyter does. Escape codes that move the cursor up are not honored.
    """
    rows = []
    for line in out.split("\n"):
        row = ""
        for part in line.split("\r"):
            row = part + row[len(part) :]
        if row:
            rows.append(row)
    return rows


def test_streamed_run_shows_a_preview_bar_as_one_row(tmp_path, monkeypatch, capsys):
    # A preview bar under the live training bar, as when train_network.py calls sample_images.
    monkeypatch.delenv("TQDM_POSITION", raising=False)
    child = _child(
        tmp_path,
        "from tqdm import tqdm\n"
        "for step in tqdm(range(3), desc='steps'):\n"
        "    if step == 1:\n"
        "        for _ in tqdm(range(30), desc='Sampling'):\n"
        "            pass\n",
    )
    code, _ = environment.run_streamed([sys.executable, str(child)], cwd=tmp_path)
    out = capsys.readouterr().out.replace("\r\n", "\n")
    assert code == 0
    assert "\x1b[A" not in out
    assert [row.split(":")[0] for row in _rows(out)] == ["Sampling", "steps"]


def test_streamed_run_passes_the_filter_to_the_child(tmp_path, monkeypatch):
    seen = {}
    real_popen = subprocess.Popen

    def spy(cmd, **kwargs):
        seen.update(kwargs.get("env") or {})
        return real_popen(cmd, **kwargs)

    monkeypatch.setenv("PYTHONWARNINGS", "default")
    # The rest of the environment reaches the child: the backend reads the tokens from it.
    monkeypatch.setenv("KT_CHILD_SENTINEL", "kept")
    monkeypatch.setattr(environment.subprocess, "Popen", spy)
    child = _child(tmp_path, "import os\nprint(os.environ['PYTHONWARNINGS'], os.environ['KT_CHILD_SENTINEL'])\n")
    code, tail = environment.run_streamed([sys.executable, str(child)], cwd=tmp_path)
    assert code == 0
    assert seen["PYTHONWARNINGS"] == "ignore::SyntaxWarning,default"
    assert tail == "ignore::SyntaxWarning,default kept"


def test_gpu_memory_reads_nvidia_smi_in_gb(monkeypatch):
    import subprocess
    import types

    answers = {}

    def run(cmd, **kwargs):
        assert cmd[:2] == ["nvidia-smi", "--query-gpu=memory.total"]
        if isinstance(answers["out"], Exception):
            raise answers["out"]
        return types.SimpleNamespace(returncode=answers.get("code", 0), stdout=answers["out"])

    monkeypatch.setattr(environment.subprocess, "run", run)
    answers.update(out="23034\n")  # an L4
    assert round(environment.gpu_memory_gb(), 1) == 22.5
    answers.update(out="15360\n15360\n")  # the first GPU counts
    assert environment.gpu_memory_gb() == 15.0
    answers.update(out="[N/A]\n")
    assert environment.gpu_memory_gb() is None
    answers.update(out="", code=9)
    assert environment.gpu_memory_gb() is None
    answers.update(out=FileNotFoundError("nvidia-smi"))
    assert environment.gpu_memory_gb() is None
    answers.update(out=subprocess.TimeoutExpired("nvidia-smi", 10))
    assert environment.gpu_memory_gb() is None

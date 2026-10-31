"""No credential is ever committed.

Scans every file we author and commit for token-shaped strings.
The patterns are built at runtime, so this file never matches itself.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from kohya_trainer import pins

from conftest import ROOT

# Used only when git cannot list the tracked files, for example in an unpacked source tree.
FALLBACK_DIRS = ("src", "scripts", "tests", "notebooks", "docs", ".github")
FALLBACK_ROOT_GLOBS = (
    "*.md",
    "*.toml",
    "*.cfg",
    "*.ini",
    "*.yml",
    "*.yaml",
    ".gitattributes",
    ".gitignore",
    ".gitmodules",
)

_HEX40 = "[0-9a-f]{40,}(?![0-9A-Za-z])"
_WANDB_KEY = "WANDB" + "_API_KEY"
TOKEN_PATTERNS = {
    "Hugging Face token": re.compile("hf" + "_" + "[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile("(?<![A-Za-z0-9])gh[pousr]" + "_" + "[A-Za-z0-9]{36}(?![A-Za-z0-9])"),
    "GitHub fine-grained token": re.compile("github" + "_pat_" + "[A-Za-z0-9]{22}_[A-Za-z0-9]{59}(?![A-Za-z0-9])"),
    # A W&B key looks like a git commit hash, so only a key handed to W&B counts.
    # Quotes can be escaped (\"), as they are inside a notebook's JSON.
    "W&B API key": re.compile(
        "(?:"
        + _WANDB_KEY
        + r"""(?:\\?["'])?\]?\s*[=:,]\s*|wandb\s+login\s+(?:--?\S+\s+)*|wandb\.login\([^)]*key\s*=\s*)(?:\\?["'])?"""
        + _HEX40
    ),
}


def find_tokens(text: str) -> list[str]:
    """Names of the token shapes found in ``text``, never the values."""
    return [name for name, pattern in TOKEN_PATTERNS.items() if pattern.search(text)]


def _git_tracked() -> list[Path] | None:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], check=False, capture_output=True, cwd=ROOT, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    names = [n for n in out.stdout.decode("utf-8", errors="replace").split("\0") if n]
    return [ROOT / n for n in names] or None


def _fallback_files() -> list[Path]:
    paths = [p for d in FALLBACK_DIRS for p in (ROOT / d).rglob("*")]
    paths += [p for g in FALLBACK_ROOT_GLOBS for p in ROOT.glob(g)]
    return paths


def files_to_scan() -> list[Path]:
    keep = set()
    for path in _git_tracked() or _fallback_files():
        rel = path.relative_to(ROOT)
        # The submodule is upstream code, and local env files are never read.
        if rel.parts[:1] == ("external",) or "__pycache__" in rel.parts or path.name.startswith(".env"):
            continue
        if path.is_file():
            keep.add(path)
    return sorted(keep)


def test_no_env_files_are_committed():
    # The scan never reads .env files, so a tracked one fails on its name alone.
    tracked = _git_tracked()
    if tracked is None:
        pytest.skip("git cannot list the tracked files here")
    env_files = [p.relative_to(ROOT).as_posix() for p in tracked if p.name.startswith(".env")]
    assert not env_files, f"remove these from git, they can hold tokens: {env_files}"


def test_scan_covers_everything_we_commit():
    scanned = {p.relative_to(ROOT).as_posix() for p in files_to_scan()}
    for expected in ("src/kohya_trainer/pins.py", "README.md", "pyproject.toml"):
        assert expected in scanned
    for optional in (".github/workflows/ci.yml", "AGENTS.md", "CONTRIBUTING.md"):
        if (ROOT / optional).is_file():
            assert optional in scanned
    assert not any(name.startswith("external/") for name in scanned)


def test_fallback_without_git_covers_the_same_places(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, b"", b""))
    scanned = {p.relative_to(ROOT).as_posix() for p in files_to_scan()}
    for expected in ("src/kohya_trainer/pins.py", "README.md", "pyproject.toml", "tests/test_no_secrets.py"):
        assert expected in scanned
    assert not any(name.startswith(("external/", ".venv/", "runs/")) for name in scanned)


def test_no_token_shaped_strings_in_tree():
    offenders = []
    for path in files_to_scan():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = find_tokens(text)
        if found:
            offenders.append(f"{path.relative_to(ROOT).as_posix()} ({', '.join(found)})")
    assert not offenders, f"token-shaped strings found in: {offenders}"


def _alnum(n: int) -> str:
    return ("Ab3" * n)[:n]


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        ("token = '" + "hf" + "_" + _alnum(34) + "'", "Hugging Face token"),
        *[("x " + "gh" + kind + "_" + _alnum(36) + " y", "GitHub token") for kind in "pousr"],
        ("github" + "_pat_" + _alnum(22) + "_" + _alnum(59), "GitHub fine-grained token"),
        ("export " + _WANDB_KEY + "=" + "0f" * 20, "W&B API key"),
        ("os.environ['" + _WANDB_KEY + "'] = \"" + "0f" * 20 + '"', "W&B API key"),
        ("!wandb login --relogin " + "0f" * 20, "W&B API key"),
        ("wandb.login(key='" + "0f" * 20 + "')", "W&B API key"),
        # The same lines with double quotes, as a notebook stores them in its JSON.
        (json.dumps({"source": 'os.environ["' + _WANDB_KEY + '"] = "' + "0f" * 20 + '"'}), "W&B API key"),
        (json.dumps({"source": 'wandb.login(key="' + "0f" * 20 + '")'}), "W&B API key"),
        (json.dumps({"source": 'os.environ.setdefault("' + _WANDB_KEY + '", "' + "0f" * 20 + '")'}), "W&B API key"),
    ],
)
def test_scanner_catches_each_shape(text, shape):
    assert shape in find_tokens(text)


@pytest.mark.parametrize(
    "text",
    [
        pins.SD_SCRIPTS_COMMIT,
        'SD_SCRIPTS_COMMIT = "' + pins.SD_SCRIPTS_COMMIT + '"',
        "git checkout " + pins.SD_SCRIPTS_COMMIT,
        "https://github.com/kohya-ss/sd-scripts/commit/" + pins.SD_SCRIPTS_COMMIT,
        "gh" + "p" + "_" + _alnum(20),
    ],
)
def test_scanner_ignores_commit_hashes_and_short_lookalikes(text):
    assert find_tokens(text) == []

"""The submodule gitlink and pins.py must always move together."""

from __future__ import annotations

import re
import subprocess

import pytest

from kohya_trainer import pins

from conftest import ROOT, SD_SCRIPTS, needs_submodule


def _gitlink() -> str | None:
    try:
        out = subprocess.run(
            ["git", "ls-tree", "HEAD", pins.SD_SCRIPTS_DIR],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
    except OSError:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    # format: "<mode> commit <sha>\t<path>"
    return out.stdout.split()[2]


def test_pinned_commit_matches_gitlink():
    gitlink = _gitlink()
    if gitlink is None:
        pytest.skip("not a git checkout")
    assert gitlink == pins.SD_SCRIPTS_COMMIT, (
        "the external/sd-scripts gitlink and pins.SD_SCRIPTS_COMMIT differ. A backend update must change both"
    )


def test_torch_policy_is_coherent():
    strict_torch = next(p for p in pins.TORCH_STRICT_PACKAGES if p.startswith("torch=="))
    major, minor = (int(x) for x in strict_torch.split("==")[1].split(".")[:2])
    assert (major, minor) >= pins.TORCH_MIN


@needs_submodule
def test_torch_min_matches_backend_readme():
    readme = (SD_SCRIPTS / "README.md").read_text(encoding="utf-8")
    found = re.search(r"PyTorch (\d+)\.(\d+)(?:\.\d+)? or later is required", readme)
    assert found, "the backend README no longer states a minimum PyTorch version. Check pins.TORCH_MIN by hand"
    assert pins.TORCH_MIN == (int(found.group(1)), int(found.group(2)))

"""Everything that depends on Colab or the machine.

Other modules must not import ``google.colab``, run pip, or assume a machine.
Tokens come from Colab Secrets or the environment, and are never logged, printed or saved.
"""

from __future__ import annotations

import codecs
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

from . import pins

TOKEN_NAMES = ("HF_TOKEN", "WANDB_API_KEY", "CIVITAI_TOKEN")


def is_colab() -> bool:
    # find_spec raises when the parent package is not installed.
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def repo_root() -> Path:
    """Root of the git clone when installed editable (src layout: two levels up)."""
    return Path(__file__).resolve().parents[2]


def sd_scripts_dir() -> Path:
    """Find the sd-scripts checkout: ``KOHYA_TRAINER_SD_SCRIPTS`` first, then ``external/sd-scripts``.

    A checkout is a folder with ``train_network.py``.
    """
    candidates = []
    override = os.environ.get("KOHYA_TRAINER_SD_SCRIPTS")
    if override:
        candidates.append(Path(override))
    candidates.append(repo_root() / pins.SD_SCRIPTS_DIR)
    for candidate in candidates:
        if (candidate / "train_network.py").is_file():
            return candidate
    raise FileNotFoundError(
        "sd-scripts checkout not found. Clone the repo with --recurse-submodules, "
        "or point KOHYA_TRAINER_SD_SCRIPTS at an sd-scripts checkout. Tried: " + ", ".join(str(c) for c in candidates)
    )


def get_secret(name: str) -> str | None:
    """Get a token from Colab Secrets, or else from the environment. Never log or print it."""
    if is_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
            if value:
                return value
        except Exception as error:  # noqa: BLE001 - any failure here falls back to the environment
            # A missing secret is normal. A secret without notebook access is a missed step.
            if type(error).__name__ == "NotebookAccessError":
                print(
                    f"note: Colab Secrets has {name}, but this notebook has no access to it. If you need it here, "
                    f"open Colab Secrets (the key icon on the left), turn on Notebook access for {name}, then run "
                    "this cell again."
                )
    return os.environ.get(name) or None


def load_tokens() -> list[str]:
    """Put the known tokens into the environment, and return the names that loaded.

    A value with a space or a hidden character inside is refused: it breaks
    every request, and some libraries then print the whole header.
    """
    loaded = []
    for name in TOKEN_NAMES:
        value = (get_secret(name) or "").strip()
        if not value:
            continue
        if any(not "!" <= char <= "~" for char in value):
            os.environ.pop(name, None)
            if name == "HF_TOKEN":
                _set_implicit_hf_token(allowed=False)
            print(
                f"warning: the {name} secret has a space, line break or hidden character inside, so it "
                f"was not loaded. Paste the token into {name} again, once, with nothing before or after "
                "it, then choose Runtime > Restart session and run this cell again."
            )
            continue
        if name == "HF_TOKEN" and _hf_token_blocked:
            # Fixed in the same session: let huggingface_hub send it again.
            _set_implicit_hf_token(allowed=True)
        os.environ[name] = value
        loaded.append(name)
    return loaded


# True after a refused HF_TOKEN turned off huggingface_hub's implicit token.
_hf_token_blocked = False
# The values huggingface_hub reads as "on" for its switches.
_ON = {"1", "ON", "YES", "TRUE"}


def _set_implicit_hf_token(allowed: bool) -> None:
    """Turn huggingface_hub's own token lookup off for a refused HF_TOKEN, or back on.

    Otherwise it reads the bad secret itself. Some versions keep it for the
    session, so the warning asks for a restart.
    """
    global _hf_token_blocked
    from huggingface_hub import constants

    switch = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").strip().upper()
    if not allowed and not _hf_token_blocked and (constants.HF_HUB_DISABLE_IMPLICIT_TOKEN or switch in _ON):
        return  # already off by the user's own choice: leave it off, never turn it back on
    if allowed:
        os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
    else:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # the training process
    constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = not allowed  # this process reads it once, at import
    _hf_token_blocked = not allowed


def mount_drive(mountpoint: str = "/content/drive") -> Path | None:
    """Mount Google Drive on Colab. Do nothing elsewhere.

    The first mount can fail while the user grants access, so "mount failed" gets one retry.
    """
    if not is_colab():
        return None
    from google.colab import drive

    try:
        drive.mount(mountpoint, timeout_ms=300_000)
    except ValueError as error:
        if not str(error).startswith("mount failed"):
            raise
        print(f"Google Drive did not mount on the first try ({error}). Trying once more.")
        drive.mount(mountpoint, timeout_ms=300_000)
    return Path(mountpoint)


DRIVE_ROOT = Path("/content/drive/MyDrive")


def drive_mounted() -> bool:
    return is_colab() and DRIVE_ROOT.is_dir()


def default_workspace() -> Path:
    """Where runs are saved: Drive when mounted, because Colab wipes /content when a session ends."""
    if drive_mounted():
        return DRIVE_ROOT / "lora"
    if is_colab():
        return Path("/content/lora")
    return Path("runs").resolve()


def scratch_dir(name: str) -> Path:
    """A fast local folder, never on Drive, for extracted datasets and latent caches."""
    root = Path("/content/scratch") if is_colab() else Path("runs").resolve() / ".scratch"
    return root / name


def total_ram_gb() -> float:
    """Total system RAM in GB, or infinity when the system does not say."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (AttributeError, ValueError, OSError):
        return float("inf")


def torch_version() -> tuple[int, int] | None:
    """The installed torch version as (major, minor), read without importing torch."""
    try:
        raw = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    parts = raw.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


def _nvidia_smi(query: str, output_format: str) -> list[str] | None:
    """The lines nvidia-smi prints for ``query``, one per GPU, or None when it gives no answer."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={output_format}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.strip().splitlines()
    return lines if out.returncode == 0 and lines else None


def gpu_name() -> str | None:
    """The first GPU's name from nvidia-smi, or None when nvidia-smi names none."""
    lines = _nvidia_smi("name", "csv,noheader")
    return lines[0] if lines else None


def gpu_memory_gb() -> float | None:
    """The first GPU's memory in GB from nvidia-smi, or None when it gives no reading."""
    lines = _nvidia_smi("memory.total", "csv,noheader,nounits")
    if not lines:
        return None
    try:
        return float(lines[0].strip()) / 1024  # nvidia-smi reports MiB
    except ValueError:
        return None


def runtime_report() -> dict:
    version = torch_version()
    return {
        "python": sys.version.split()[0],
        "torch": ".".join(map(str, version)) if version else None,
        "gpu": gpu_name(),
        "colab": is_colab(),
    }


# Backend log lines that repeat once per module. Only the first shows, and the repeats are counted.
REPEATED_LINES = ("LoKr: lora_dim",)

# Backend log lines that tell the user nothing: httpx's line for each Hub
# request, and onnxruntime's CPU note. A failed request still shows its error.
HIDDEN_LINES = ("HTTP Request: ", "VerifyEachNodeIsAssignedToAnEp")

# Upstream scripts have invalid escapes in strings, which newer Pythons warn about.
CHILD_WARNING_FILTERS = "ignore::SyntaxWarning"

# tqdm draws a bar opened under a live one (a preview under the training bar)
# on the next row and moves back with an escape code that Colab ignores, so
# each preview step printed a new line. Row 0 for every bar stops that.
# tqdm reads TQDM_POSITION from the environment as a number since 4.66.1.
CHILD_TQDM_POSITION = "0"

_LINE_BREAKS = re.compile(r"(\r\n|\r|\n)")


class _LineFilter:
    """Hide HIDDEN_LINES and the repeats of REPEATED_LINES in a stream of text.

    Text passes a full line at a time. A line with a carriage return is a
    progress bar, and passes as it arrives. The wrapped rest of a matched
    line, indented to the marker's column, is shown or hidden with it.
    """

    def __init__(self, repeated: tuple[str, ...] = REPEATED_LINES, hidden: tuple[str, ...] = HIDDEN_LINES):
        self.repeated = repeated
        self.always_hidden = hidden
        self.held = ""  # the start of a line that has no line break yet
        self.cr = ""  # a "\r" at the end of a read, which can start a "\r\n"
        self.live = False  # this line is a progress bar, written as it arrives
        self.seen: set[str] = set()
        self.last: str | None = None  # the marker of the last full line, if any
        self.col = 0  # where the marker starts in that line
        self.last_hidden = False
        self.repeats = 0  # repeats hidden since the last note

    def feed(self, text: str) -> str:
        text, self.cr = self.cr + text, ""
        if text.endswith("\r"):
            text, self.cr = text[:-1], "\r"
        out = []
        for piece in _LINE_BREAKS.split(text):
            if piece in ("\n", "\r\n"):
                out.append(piece if self.live else self._line(self.held + piece))
                self.held, self.live = "", False
            elif piece == "\r":
                if not self.live:
                    out.append(self._note() + self.held)
                    self.held, self.live, self.last = "", True, None
                out.append("\r")
            elif piece:
                if self.live:
                    out.append(piece)
                else:
                    self.held += piece
        return "".join(out)

    def finish(self) -> str:
        """What is left when the stream ends: a last line without a line break, and the hidden count."""
        rest = self.feed("") + self._note() + self.held + self.cr
        self.held = self.cr = ""
        return rest

    def _line(self, line: str) -> str:
        marker = next((m for m in self.always_hidden + self.repeated if m in line), None)
        if marker is None:
            indent = len(line) - len(line.lstrip(" \t"))
            if self.last is not None and self.col and indent >= self.col and line.strip():
                return "" if self.last_hidden else line
            self.last = None
            return self._note() + line
        self.last, self.col = marker, line.index(marker)
        if marker in self.always_hidden:  # no count for these
            self.last_hidden = True
            return ""
        self.last_hidden = marker in self.seen
        if self.last_hidden:
            self.repeats += 1
            return ""
        self.seen.add(marker)
        return self._note() + line

    def _note(self) -> str:
        if not self.repeats:
            return ""
        count, self.repeats = self.repeats, 0
        if count == 1:
            return "(1 more line like the one above was hidden)\n"
        return f"({count} more lines like the one above were hidden)\n"


def _child_env() -> dict[str, str]:
    """This process's environment, plus CHILD_WARNING_FILTERS and CHILD_TQDM_POSITION. The user's own values win."""
    env = dict(os.environ)
    current = env.get("PYTHONWARNINGS", "")
    env["PYTHONWARNINGS"] = f"{CHILD_WARNING_FILTERS},{current}" if current else CHILD_WARNING_FILTERS
    env.setdefault("TQDM_POSITION", CHILD_TQDM_POSITION)
    return env


def run_streamed(cmd: list[str], cwd: str | Path) -> tuple[int, str]:
    """Run a child process and copy its filtered output to sys.stdout, which is all a Colab cell shows.

    Returns the exit code and the end of the output. Colab's stop button
    stops the child too. The training script under ``accelerate launch``
    stops only at its next output, when it finds its pipe closed.
    """
    process = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_child_env())
    tail: deque[str] = deque(maxlen=40)
    # Keeps a character split between two reads whole.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    quiet = _LineFilter()
    fd = process.stdout.fileno()
    try:
        while True:
            chunk = os.read(fd, 4096)  # partial reads keep tqdm's \r updates live
            text = quiet.feed(decoder.decode(chunk, final=not chunk))
            if not chunk:
                text += quiet.finish()
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                tail.append(text)
            if not chunk:
                break
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    return returncode, "".join(tail).strip()[-3000:]


def _pip(*args: str, quiet: bool = True, cwd: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(args)
    subprocess.run(cmd, check=True, cwd=cwd)


def install(strict_torch: bool = False, quiet: bool = True) -> dict:
    """Install the training backend into the current kernel.

    It keeps a torch of at least TORCH_MIN unless ``strict_torch=True``. The
    packages come from the submodule's requirements.txt. It raises if torch
    no longer matches the torch that torchvision was built for.
    """
    version = torch_version()
    if strict_torch or version is None or version < pins.TORCH_MIN:
        _pip(*pins.TORCH_STRICT_PACKAGES, "--index-url", pins.TORCH_STRICT_INDEX, quiet=quiet)
    backend = sd_scripts_dir()
    # requirements.txt has a relative "-e .", so pip runs from the backend folder.
    _pip("-r", "requirements.txt", quiet=quiet, cwd=str(backend))
    mismatch = torch_pair_mismatch()
    if mismatch:
        installed, wanted = mismatch
        raise RuntimeError(
            f"torch {installed} is installed, but torchvision was built for torch {wanted}. "
            "Installing the backend replaced the torch this runtime came with, so training will fail. "
            "In Colab, choose Runtime > Disconnect and delete runtime, then run this cell again. "
            f"If that does not help, please report it at {pins.REPO_URL}/issues."
        )
    ensure_accelerate_config()
    return runtime_report()


def _installed(name: str) -> str | None:
    """The full installed version of a package, local tag included (for example 2.8.0+cu126)."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_pair_mismatch() -> tuple[str, str] | None:
    """(installed torch, torch that torchvision needs) when the two disagree, else None."""
    try:
        requires = importlib.metadata.requires("torchvision") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    wanted = None
    for requirement in requires:
        match = re.match(r"torch\s*\(?\s*==\s*([^\s;,)]+)", requirement)
        if match:
            wanted = match.group(1)
            break
    if wanted is None:
        return None
    installed = _installed("torch")
    if installed is None:
        return ("(none)", wanted)
    # A pin without a local tag (2.8.0) accepts any build (2.8.0+cu126).
    have = installed if "+" in wanted else installed.split("+")[0]
    return None if have == wanted else (installed, wanted)


def ensure_tagger_deps(quiet: bool = True) -> None:
    """Install onnx and onnxruntime, which the tagger needs and upstream keeps optional."""
    if importlib.util.find_spec("onnx") is None:
        _pip("onnx", quiet=quiet)
    if importlib.util.find_spec("onnxruntime") is None:
        _pip("onnxruntime-gpu" if gpu_name() else "onnxruntime", quiet=quiet)


def ensure_wandb(quiet: bool = True) -> None:
    """Install wandb before the launch. It is not in the backend requirements."""
    if importlib.util.find_spec("wandb") is None:
        _pip("wandb", quiet=quiet)


def wandb_login_available() -> bool:
    """Whether the training process can run wandb.init without a prompt.

    An offline WANDB_MODE, WANDB_API_KEY, or a wandb entry in ~/.netrc counts.
    """
    if os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled", "dryrun"):
        return True
    if os.environ.get("WANDB_API_KEY"):
        return True
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        try:
            if path.is_file() and "api.wandb.ai" in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def ensure_accelerate_config() -> None:
    """Write a default single-GPU accelerate config if none exists, in a subprocess so the kernel skips torch."""
    config = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    if config.exists():
        return
    subprocess.run(
        [sys.executable, "-c", "from accelerate.utils import write_basic_config; write_basic_config()"],
        check=True,
    )

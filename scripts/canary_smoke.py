"""The Colab canary: run the user's path on today's unpinned packages.

A failed scheduled run means something outside our code changed: a Colab update,
a package conflict or a model removed from the Hub. Stages: install and pip check,
backend files and imports, dry-run plans for every route and base, listed Hub files.
tests/test_canary.py tests the stages that need no network.
"""

from __future__ import annotations

import dataclasses
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kohya_trainer import config, environment, models, registry, tagging, train


def png_1px() -> bytes:
    """A valid 1x1 grayscale PNG, made with the standard library."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # 1x1, 8-bit grayscale
    pixels = zlib.compress(b"\x00\x80")  # row filter 0 + one gray pixel
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def make_dataset(target: Path, count: int = 4) -> Path:
    """A tiny captioned dataset, with the images in the folder itself so the trainer counts them."""
    target.mkdir(parents=True, exist_ok=True)
    image = png_1px()
    for i in range(count):
        (target / f"img_{i}.png").write_bytes(image)
        (target / f"img_{i}.txt").write_text("1girl, canary, solo", encoding="utf-8", newline="\n")
    return target


def entrypoint_modules() -> list[str]:
    return sorted({Path(route.entrypoint).stem for route in registry.ROUTES.values()})


def preset_network_modules() -> list[str]:
    """The network_module values the bundled presets name, sorted and without repeats."""
    presets = resources.files("kohya_trainer") / "presets"
    names = sorted(entry.name for entry in presets.iterdir() if entry.name.endswith(".toml"))
    modules = {config.load_preset(name).get("network_module") for name in names}
    return sorted(module for module in modules if module)


def backend_files() -> list[Path]:
    """The backend files our commands use besides the entrypoints: the tagger script and the network modules."""
    modules = sorted({*train.NETWORK_MODULES.values(), *preset_network_modules()})
    files = [Path("finetune") / "tag_images_by_wd14_tagger.py"]
    files += [Path(module.replace(".", "/") + ".py") for module in modules]
    return files


def check_backend_files() -> None:
    backend = environment.sd_scripts_dir()
    for relative in backend_files():
        if not (backend / relative).is_file():
            raise FileNotFoundError(f"{relative} missing from the pinned backend")


def import_entrypoints() -> None:
    """Import each backend entrypoint in a new interpreter: the closest check to a training start without a GPU."""
    backend = environment.sd_scripts_dir()
    for module in entrypoint_modules():
        print(f"import {module} ...", flush=True)
        subprocess.run([sys.executable, "-c", f"import {module}"], check=True, cwd=str(backend))


def check_training_names() -> None:
    """Check that each listed optimizer imports and that the installed transformers knows each scheduler name."""
    import importlib

    from transformers.optimization import SchedulerType

    known = {item.value for item in SchedulerType}
    unknown = sorted(set(train.TRANSFORMERS_SCHEDULERS) - known)
    if unknown:
        raise RuntimeError(f"schedulers the plan accepts but transformers does not know: {unknown}")
    for name in train.OPTIMIZER_NAMES.values():
        module, _, attribute = name.rpartition(".")
        if not hasattr(importlib.import_module(module), attribute):
            raise RuntimeError(f"optimizer {name} is listed, but {module} has no {attribute}")
        print(f"optimizer {name}: ok", flush=True)


def dry_run_plans(workspace: Path) -> list[train.LaunchPlan]:
    """Run build_plan for every route and listed base, with empty model files and a made-up dataset.

    Planning never reads model files. A ConvRot INT8 route also plans with its flags, as a pasted base gets them."""
    backend = environment.sd_scripts_dir()
    dataset_dir = make_dataset(workspace / "dataset")
    fake_models = workspace / "models"
    fake_models.mkdir(parents=True, exist_ok=True)
    plans = []
    for route in registry.ROUTES.values():
        if not (backend / route.entrypoint).is_file():
            raise FileNotFoundError(f"entrypoint {route.entrypoint} missing from the pinned backend")
        for base in route.base_models:
            checkpoint = fake_models / f"{route.key}.{base.key}.checkpoint.safetensors"
            checkpoint.touch()
            vae = None
            if route.default_vae:
                vae = fake_models / f"{route.key}.{base.key}.vae.safetensors"
                vae.touch()
            text_encoder = None
            if route.text_encoder_arg:
                text_encoder = fake_models / f"{route.key}.{base.key}.te.safetensors"
                text_encoder.touch()
            variants = [("", base)]
            if route.convrot_int8:
                variants.append(("-convrot", dataclasses.replace(base, convrot_flags=models.CONVROT_FLAGS)))
            for suffix, planned in variants:
                plan = train.build_plan(
                    route,
                    planned,
                    checkpoint_path=checkpoint,
                    vae_path=vae,
                    text_encoder_path=text_encoder,
                    dataset_dir=dataset_dir,
                    workspace=workspace / f"{route.key}-{base.key}{suffix}",
                    project_name=f"canary_{route.key}",
                )
                assert plan.config_file.is_file() and plan.dataset_config_file.is_file()
                plans.append(plan)
    return plans


def registry_artifacts() -> list[tuple[str, str | None]]:
    """Every (repo_id, filename) in the registry, ConvRot INT8 files included, without repeats."""
    specs: list[tuple[str, str | None]] = []
    for route in registry.ROUTES.values():
        for spec in (*route.base_models, route.default_vae, route.default_text_encoder):
            if spec is None:
                continue
            files = [spec.filename] + ([spec.int8_filename] if spec.int8_filename else [])
            for filename in files:
                if (spec.repo_id, filename) not in specs:
                    specs.append((spec.repo_id, filename))
    return specs


def curated_artifacts() -> list[tuple[str, str | None]]:
    """Everything a notebook can download from the Hub: the registry's files, the files a trainer
    fetches when it starts, and the WD taggers by repo id."""
    specs = registry_artifacts()
    for route in registry.ROUTES.values():
        for spec in route.runtime_files:
            if spec not in specs:
                specs.append(spec)
    for repo_id in tagging.TAGGERS:
        if (repo_id, None) not in specs:
            specs.append((repo_id, None))
    return specs


def _exists(api, repo_id: str, filename: str | None) -> bool | None:
    """Return True or False, or None when the Hub gives no answer after a retry."""
    from huggingface_hub.errors import GatedRepoError

    for attempt in (1, 2):
        try:
            return api.file_exists(repo_id, filename) if filename else api.repo_exists(repo_id)
        except GatedRepoError:
            return True  # the repo exists but asks the user to accept a license
        # Any other error is a short Hub or network problem, so retry once.
        except Exception as error:  # noqa: BLE001
            failure = type(error).__name__
            if attempt == 1:
                time.sleep(10)
    print(f"  inconclusive {repo_id}: {failure}", flush=True)
    return None


def check_artifacts() -> tuple[list[str], list[str]]:
    """Check that every listed Hub file exists. Returns (missing, inconclusive)."""
    from huggingface_hub import HfApi  # needs the network, so imported only here

    api = HfApi()
    missing: list[str] = []
    inconclusive: list[str] = []
    for repo_id, filename in curated_artifacts():
        label = f"{repo_id}: {filename}" if filename else repo_id
        verdict = _exists(api, repo_id, filename)
        if verdict is None:
            inconclusive.append(label)
        elif verdict:
            print(f"  ok       {label}", flush=True)
        else:
            print(f"  missing  {label}", flush=True)
            missing.append(label)
    return missing, inconclusive


def main() -> int:
    print("== stage 1: backend install (keeping the preinstalled torch) ==", flush=True)
    print("runtime:", environment.install(), flush=True)
    # The 1-click path installs these when it tags, so the conflict check covers them.
    environment.ensure_tagger_deps()
    # pip exits 0 on conflicts it did not have to resolve. pip check fails on all of them.
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    print("== stage 2: backend (launcher, files, entrypoint imports) ==", flush=True)
    # The subcommand execute() uses. accelerate has no top-level --version (1.6.0 exits 2 on it).
    subprocess.run(["accelerate", "launch", "--help"], check=True, capture_output=True)
    print("accelerate launch: ok", flush=True)
    check_backend_files()
    import_entrypoints()
    check_training_names()

    print("== stage 3: dry-run plans for every route and base ==", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        plans = dry_run_plans(Path(tmp))
    print(f"planned {len(plans)} route/base combinations", flush=True)

    print("== stage 4: listed Hub files ==", flush=True)
    missing, inconclusive = check_artifacts()
    if missing:
        print(f"canary failed: {len(missing)} file(s) gone from the Hub: {missing}")
        return 1
    if inconclusive:
        print(f"note: {len(inconclusive)} Hub check(s) inconclusive because of network trouble: {inconclusive}")
        if len(inconclusive) == len(curated_artifacts()):
            print("canary failed: every Hub check gave an error, so the Hub API path is broken")
            return 1

    print("canary passed: the Colab user path works with today's packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Find and download base models from the Hugging Face Hub, Civitai or a direct link.

HF_TOKEN and CIVITAI_TOKEN are read from the environment (environment.load_tokens).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import re
from pathlib import Path

from . import environment, registry
from .registry import BaseModel, ModelRoute

USER_AGENT = "kohya-trainer (github.com/Linaqruf/kohya-trainer)"
DOWNLOAD_CACHE = Path.home() / ".cache" / "kohya_trainer" / "models"

# Civitai "baseModel" labels for each route. Another label gets a warning.
CIVITAI_FAMILIES = {
    "sd15": ("SD 1.5",),
    "sdxl": ("SDXL 1.0",),
    "illustrious": ("Illustrious", "NoobAI", "SDXL 1.0"),
}

# Anchored, so an HF URL with ?ref=civitai.com is not taken for Civitai.
_CIVITAI_HOST = re.compile(r"^(?:https?://)?(?:www\.)?civitai\.com/", re.IGNORECASE)
_CIVITAI_DOWNLOAD_URL = re.compile(r"civitai\.com/api/download/models/(\d+)", re.IGNORECASE)
_CIVITAI_API_VERSION_URL = re.compile(r"civitai\.com/api/v1/model-versions/(\d+)", re.IGNORECASE)
_CIVITAI_MODEL_URL = re.compile(r"civitai\.com/(?:api/v1/)?models/(\d+)", re.IGNORECASE)
_CIVITAI_VERSION_PARAM = re.compile(r"[?&]modelVersionId=(\d+)", re.IGNORECASE)
_HF_URL = re.compile(
    r"huggingface\.co/([\w.-]+/[\w.-]+)(?:/(?:blob|resolve)/[^/]+/(.+?))?/?(?:[?#].*)?$", re.IGNORECASE
)
# Civitai has no v-pred field, so the version's name and description are the only clue.
_VPRED_HINT = re.compile(r"v[-_ ]?pred|zero.?terminal|ztsnr", re.IGNORECASE)


def custom_base(spec: str, route_key: str = "") -> BaseModel:
    """Turn a pasted model spec into a BaseModel.

    Takes ``user/repo`` or ``user/repo:file``, an HF URL, a Civitai URL, or a
    direct https link to a .safetensors file. A pasted base is never v-pred on its own.
    """
    spec = spec.strip()
    if _CIVITAI_HOST.match(spec):
        _check_civitai_id(spec)
        return _civitai_base(spec, route_key)
    hf = _HF_URL.search(spec)
    if hf:
        repo_id, filename = hf.group(1), hf.group(2)
        if repo_id.split("/")[0].lower() in {"datasets", "spaces", "collections"}:
            raise ValueError(f"only model repos can be a training base, got {spec!r}")
        return BaseModel(key=spec, display_name=repo_id, repo_id=repo_id, filename=filename or None)
    if spec.startswith(("http://", "https://")):
        filename = spec.rsplit("/", 1)[-1].split("?")[0]
        if not filename.endswith(".safetensors"):
            raise ValueError(
                f"direct links must point at a .safetensors file, got {spec!r}. "
                "For a model page, use a Hugging Face or Civitai URL."
            )
        return BaseModel(key=spec, display_name=filename, repo_id="", filename=filename, download_url=spec)
    repo_id, _, filename = spec.partition(":")
    if repo_id.count("/") != 1:
        raise ValueError(
            f"expected a Hugging Face repo id like 'user/repo', an HF URL, or a Civitai model URL. Got {spec!r}."
        )
    return BaseModel(key=spec, display_name=spec, repo_id=repo_id, filename=filename or None)


def _check_civitai_id(spec: str) -> None:
    """Refuse a Civitai link with no model or version id in it."""
    if not (
        _CIVITAI_DOWNLOAD_URL.search(spec) or _CIVITAI_API_VERSION_URL.search(spec) or _CIVITAI_MODEL_URL.search(spec)
    ):
        raise ValueError(
            f"no model id in that Civitai link ({spec!r}). Paste the model page URL, like civitai.com/models/<id>"
        )


def _civitai_headers() -> dict:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("CIVITAI_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _civitai_json(url: str) -> dict:
    import requests

    response = requests.get(url, headers=_civitai_headers(), timeout=60)
    if response.status_code == 404:
        raise ValueError(f"Civitai returned 404 for {url}. The model was deleted, or the link has a typo.")
    response.raise_for_status()
    return response.json()


def _score_civitai_file(file: dict) -> tuple:
    meta = file.get("metadata") or {}
    return (meta.get("format") == "SafeTensor", meta.get("fp") == "fp16", meta.get("size") == "pruned")


def _pick_civitai_file(version: dict, model_name: str) -> dict:
    """The file the website picks: the primary Model file, or else the best SafeTensor."""
    candidates = [f for f in version.get("files") or [] if f.get("type") == "Model"]
    if not candidates:
        raise ValueError(f"no model-weights file on Civitai version {version.get('id')} of {model_name!r}")
    primaries = [f for f in candidates if f.get("primary")]
    chosen = (
        candidates[0]
        if len(candidates) == 1
        else (primaries[0] if primaries else max(candidates, key=_score_civitai_file))
    )
    if (chosen.get("metadata") or {}).get("format") == "PickleTensor":
        raise ValueError(
            f"the Civitai file for {model_name!r} is a pickle checkpoint. Only SafeTensor downloads "
            "are supported. Pick a SafeTensor version on the model page."
        )
    return chosen


def _civitai_version(spec: str) -> tuple[dict, str, str]:
    """The version a Civitai URL names, with its model's name and type."""
    direct = _CIVITAI_DOWNLOAD_URL.search(spec) or _CIVITAI_API_VERSION_URL.search(spec)
    if direct:
        version = _civitai_json(f"https://civitai.com/api/v1/model-versions/{direct.group(1)}")
        model = version.get("model") or {}
        model_name = model.get("name") or f"civitai version {direct.group(1)}"
        model_type = model.get("type") or "Checkpoint"
        return version, model_name, model_type
    model_id = _CIVITAI_MODEL_URL.search(spec).group(1)
    data = _civitai_json(f"https://civitai.com/api/v1/models/{model_id}")
    model_name = data.get("name") or f"civitai model {model_id}"
    model_type = data.get("type") or "Checkpoint"
    versions = data.get("modelVersions") or []
    if not versions:
        raise ValueError(f"Civitai model {model_id} ({model_name!r}) has no published versions")
    wanted = _CIVITAI_VERSION_PARAM.search(spec)
    if wanted:
        version = next((v for v in versions if str(v.get("id")) == wanted.group(1)), None)
        if version is None:
            raise ValueError(f"modelVersionId {wanted.group(1)} is not a version of Civitai model {model_id}")
    else:
        # The version the model page shows first.
        version = versions[0]
    return version, model_name, model_type


def _check_civitai_type(model_name: str, model_type: str, want: str) -> None:
    """Refuse a Civitai model of another type than ``want``."""
    if want == "Checkpoint" and model_type != "Checkpoint":
        raise ValueError(
            f"{model_name!r} is a Civitai {model_type}, not a base checkpoint. This field wants the model you train on."
        )
    if want == "LORA" and model_type not in ("LORA", "LoCon"):
        raise ValueError(
            f"{model_name!r} is a Civitai {model_type}. network_weights wants a finished LoRA or LoCon to continue from."
        )


def _civitai_family_warning(version: dict, model_name: str, route_key: str, want: str) -> None:
    """Warn when Civitai tags the model for another family than this route's."""
    family = version.get("baseModel") or ""
    expected = CIVITAI_FAMILIES.get(route_key)
    if expected and family and family not in expected:
        noun = "checkpoint" if want == "Checkpoint" else "LoRA"
        print(
            f"warning: {model_name!r} is tagged '{family}' on Civitai, but this notebook trains "
            f"{' / '.join(expected)} models. Check that the {noun} you pasted is the right one."
        )


def _civitai_vpred_warning(version: dict, route_key: str) -> None:
    """Warn when a pasted checkpoint looks like v-prediction."""
    hints = f"{version.get('name') or ''} {version.get('description') or ''}"
    if _VPRED_HINT.search(hints):
        route = registry.ROUTES.get(route_key)
        if route is not None and route.custom_vpred_option:
            way = "tick custom_base_is_vpred under custom_base"
        else:
            way = (
                "add 'v_parameterization': True, 'zero_terminal_snr': True, "
                "'scale_v_pred_loss_like_noise_pred': True, 'debiased_estimation_loss': True and "
                "'sample_sampler': 'euler' to config_overrides"
            )
        print(
            "warning: this checkpoint looks like v-prediction, from its Civitai version name or description. "
            f"Pasted models do not get the v-pred flags on their own. If it really is v-pred, {way}."
        )


def _civitai_base(spec: str, route_key: str, want: str = "Checkpoint") -> BaseModel:
    """Turn a Civitai URL into a downloadable file. ``want`` is "Checkpoint" or "LORA"."""
    version, model_name, model_type = _civitai_version(spec)
    _check_civitai_type(model_name, model_type, want)
    file = _pick_civitai_file(version, model_name)

    _civitai_family_warning(version, model_name, route_key, want)
    if want == "Checkpoint":
        _civitai_vpred_warning(version, route_key)
    display = f"{model_name} — {version.get('name') or ''}".strip(" —")
    return BaseModel(
        key=spec,
        display_name=display,
        repo_id=f"civitai:{version.get('id')}",
        filename=file["name"],
        download_url=file.get("downloadUrl") or f"https://civitai.com/api/download/models/{version['id']}",
    )


def resolve_filename(repo_id: str) -> str:
    """Pick the one top-level .safetensors file of a repo, or raise with the list of files."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id)
    candidates = [f for f in files if f.endswith(".safetensors") and "/" not in f]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"cannot pick a checkpoint in {repo_id!r} (top-level .safetensors found: {candidates or 'none'}). "
        f"Name one in the spec, as '{repo_id}:<filename>'"
    )


def download(base: BaseModel) -> Path:
    """Download a model, or reuse the cached copy, and return its path.

    Hub files go to a local_dir as real files. The backend misreads the
    default Hub cache's symlinks as a Diffusers repo path (_load_target_model in
    library/sdxl_train_util.py).
    """
    if base.download_url:
        return _download_url(base)
    from huggingface_hub import hf_hub_download

    filename = base.filename or resolve_filename(base.repo_id)
    return Path(hf_hub_download(base.repo_id, filename, local_dir=DOWNLOAD_CACHE / _repo_tag(base.repo_id)))


def _repo_tag(repo_id: str) -> str:
    """The cache folder name for a repo id: every run of other characters becomes one "-"."""
    return re.sub(r"[^\w.-]+", "-", repo_id)


def _cache_path(base: BaseModel) -> Path:
    """The cache path for a model, in a folder per source, because models share file names.

    Only the last part of the file name is kept, so remote data cannot write outside the cache.
    """
    if base.repo_id:
        tag = _repo_tag(base.repo_id)
    else:
        tag = hashlib.sha256(base.download_url.encode("utf-8")).hexdigest()[:16]
    return DOWNLOAD_CACHE / tag / Path(base.filename).name


def _download_url(base: BaseModel) -> Path:
    """Download a direct or Civitai link into the local cache. A file at the final path is complete."""
    import requests

    target = _cache_path(base)
    if target.is_file():
        print(f"cached: {target}")
        return target
    civitai = bool(_CIVITAI_HOST.match(base.download_url))
    headers = _civitai_headers() if civitai else {"User-Agent": USER_AGENT}
    with requests.Session() as session:
        if civitai:
            # A file that needs a login answers with a JSON 401 here. After the
            # redirect, requests drops the token, but the signed CDN URL needs no
            # login. Other links skip this request,
            # which can use up a single-use signed URL.
            first = session.get(base.download_url, headers=headers, allow_redirects=False, stream=True, timeout=60)
            if first.status_code == 401:
                # A body that is not JSON still gets the hint.
                try:
                    message = first.json().get("message")
                except (ValueError, AttributeError, requests.RequestException):
                    message = None
                detail = message if isinstance(message, str) else ""
                raise PermissionError(
                    f"this download requires a Civitai login{': ' + detail if detail else ''}. "
                    "Create an API key at civitai.com/user/account (API Keys section) and add it as the "
                    "CIVITAI_TOKEN Colab secret, then run this cell again."
                )
            first.close()
        response = session.get(base.download_url, headers=headers, allow_redirects=True, stream=True, timeout=60)
        _save(response, target)
    return target


def download_file(url: str, target: Path) -> Path:
    """Download a direct https link to ``target``, or reuse the complete file already there."""
    import requests

    target = Path(target)
    if target.is_file():
        print(f"cached: {target}")
        return target
    with requests.Session() as session:
        response = session.get(url, headers={"User-Agent": USER_AGENT}, allow_redirects=True, stream=True, timeout=60)
        _save(response, target)
    return target


def _save(response, target: Path) -> None:
    from tqdm.auto import tqdm

    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {target.name} ...")
    tmp = target.with_suffix(target.suffix + ".tmp")
    declared = int(response.headers.get("Content-Length") or 0)
    received = 0
    with (
        open(tmp, "wb") as handle,
        tqdm(total=declared or None, unit="B", unit_scale=True, unit_divisor=1024) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
            received += len(chunk)
            bar.update(len(chunk))
    if declared and received < declared:
        tmp.unlink(missing_ok=True)
        raise OSError(f"download truncated: got {received} of {declared} bytes. Run the cell again to retry.")
    os.replace(tmp, target)


def resolve_network_weights(spec: str, route_key: str = "") -> Path:
    """Turn the network_weights field (a path, an HF repo or URL, a Civitai URL or a direct link) into a local path."""
    spec = spec.strip()
    if _CIVITAI_HOST.match(spec):
        _check_civitai_id(spec)
        return download(_civitai_base(spec, route_key, want="LORA"))
    if spec.startswith(("http://", "https://")) or _HF_URL.search(spec):
        return download(custom_base(spec))
    local = Path(spec).expanduser()
    if local.is_file():
        return local.resolve()
    # "output/lora.safetensors" is a missing local file, not a repo id.
    remote_spec = re.fullmatch(r"[\w.-]+/[\w.-]+(?::[^/\\]+)?", spec)
    if remote_spec and (":" in spec or not spec.lower().endswith((".safetensors", ".ckpt", ".pt"))):
        return download(custom_base(spec))
    hint = ""
    if spec.startswith("/content/drive") and not Path("/content/drive").is_dir():
        hint = ". Google Drive is not mounted. Tick mount_google_drive, or run environment.mount_drive()."
    raise FileNotFoundError(f"network_weights file not found: {spec}{hint}")


CONVROT_CHOICES = ("auto", "on", "off")
# "auto" turns ConvRot INT8 on below these. Upstream gives MOSS-Image's bf16
# LoRA a peak of 15.3 GB of GPU memory and 17.2 GB of RAM at batch 1
# (docs/moss_image_train_network.md).
CONVROT_AUTO_GPU_GB = 20
CONVROT_AUTO_RAM_GB = 24
# The backend flags for the DiT and the text encoder, and their names in the notes.
CONVROT_FLAGS = ("convrot_int8", "convrot_int8_text_encoder")
MODEL_NAMES = ("DiT", "text encoder")


@dataclasses.dataclass(frozen=True)
class ConvRotChoice:
    """Whether the DiT and the text encoder train on ConvRot INT8 weights, and why."""

    dit: bool = False
    text_encoder: bool = False
    note: str = ""


_READ = object()


def convrot_choice(
    route: ModelRoute,
    form_value: str = "auto",
    config_overrides: dict | None = None,
    *,
    gpu_memory_gb=_READ,
    ram_gb=_READ,
) -> ConvRotChoice:
    """Decide ConvRot INT8 for a run. Both the downloads and the flags follow this.

    A flag in ``config_overrides`` wins for its model. Otherwise "on" picks
    both, "off" neither, and "auto" both when the GPU memory or the RAM is short.
    """
    if not route.convrot_int8:
        return ConvRotChoice()
    value = str(form_value or "auto").strip().lower()
    if value not in CONVROT_CHOICES:
        raise ValueError(f"convrot_int8 is {form_value!r}. It takes auto, on or off.")
    if value == "auto":
        gpu = environment.gpu_memory_gb() if gpu_memory_gb is _READ else gpu_memory_gb
        ram = environment.total_ram_gb() if ram_gb is _READ else ram_gb
        wanted, why = _convrot_auto(gpu, ram)
    else:
        wanted, why = value == "on", "as convrot_int8 asks"
    picked, reasons = _convrot_per_model(config_overrides or {}, wanted, why)
    return ConvRotChoice(picked["convrot_int8"], picked["convrot_int8_text_encoder"], _convrot_note(picked, reasons))


def _convrot_auto(gpu: float | None, ram: float) -> tuple[bool, str]:
    """What "auto" picks for this GPU memory and RAM in GB, and the reason the note gives."""
    short_gpu = gpu is not None and gpu < CONVROT_AUTO_GPU_GB
    short_ram = ram < CONVROT_AUTO_RAM_GB
    if short_gpu:
        why = f"because this GPU has {gpu:.0f} GB of memory"
    elif short_ram:
        why = f"because this machine has {ram:.0f} GB of RAM"
    elif gpu is not None and math.isfinite(ram):
        why = f"as this GPU has {gpu:.0f} GB and the machine has {ram:.0f} GB of RAM"
    elif math.isfinite(ram):
        why = f"as the machine has {ram:.0f} GB of RAM (no GPU memory reading was found)"
    elif gpu is not None:
        why = f"as this GPU has {gpu:.0f} GB (the RAM could not be read)"
    else:
        why = "as neither the GPU memory nor the RAM could be read"
    return short_gpu or short_ram, why


def _convrot_per_model(overrides: dict, wanted: bool, why: str) -> tuple[dict, dict]:
    """Each model's choice and its reason, by flag. None in config_overrides means off."""
    picked, reasons = {}, {}
    for flag in CONVROT_FLAGS:
        overridden = flag in overrides
        given = overrides.get(flag)
        if overridden and not isinstance(given, bool) and given is not None:
            raise ValueError(f"{flag} in config_overrides is {given!r}. It takes True, False or None.")
        picked[flag] = bool(given) if overridden else wanted
        reasons[flag] = "as config_overrides asks" if overridden else why
    return picked, reasons


def _convrot_note(picked: dict, reasons: dict) -> str:
    """The note that says what ConvRot INT8 does in this run, and why."""
    if len({(picked[flag], reasons[flag]) for flag in CONVROT_FLAGS}) > 1:
        parts = [
            f"{'on' if picked[flag] else 'off'} for the {name} ({reasons[flag]})"
            for flag, name in zip(CONVROT_FLAGS, MODEL_NAMES)
        ]
        return f"ConvRot INT8 is {' and '.join(parts)}."
    reason = reasons["convrot_int8"]
    state = f"ConvRot INT8 is {'on' if picked['convrot_int8'] else 'off'}{',' if reason.startswith('as ') else ''}"
    if picked["convrot_int8"]:
        return f"{state} {reason}."
    return f"{state} {reason}, so the bf16 files are used, the reference quality."


def _convrot_file(model: BaseModel, wanted: bool, flag: str) -> tuple[BaseModel, tuple[str, ...]]:
    """The file to download for ``model``, and the flag the backend then needs.

    Without an int8 file in the repo, the flag makes the backend quantize the model while it loads.
    """
    if not wanted:
        return model, ()
    if model.int8_filename:
        return dataclasses.replace(model, filename=model.int8_filename), ()
    return model, (flag,)


def _convrot_effects(choice: ConvRotChoice, dit_flags: tuple, te_flags: tuple) -> list[str]:
    """The sentences that say what the ConvRot choice does to this run's files."""
    effects = []
    models_used = ((MODEL_NAMES[0], choice.dit, dit_flags), (MODEL_NAMES[1], choice.text_encoder, te_flags))
    int8_files = [name for name, wanted, flags in models_used if wanted and not flags]
    if len(int8_files) == 2:
        effects.append(
            "The int8 files are about half the size and GPU memory of the bf16 ones, and the LoRA loads like any other."
        )
    elif int8_files:
        effects.append(f"The {int8_files[0]}'s int8 file is about half the size and GPU memory of the bf16 one.")
    converted = [name for name, _, flags in models_used if flags]
    if converted:
        verbs = "downloads as given and turns" if len(converted) == 1 else "download as given and turn"
        effects.append(
            f"The {' and the '.join(converted)} {verbs} into int8 while loading, if not int8 already, "
            "which saves GPU memory."
        )
    return effects


def prepare_base(
    route: ModelRoute,
    base_key: str | None = None,
    custom: str | BaseModel = "",
    convrot_int8: str = "auto",
    config_overrides: dict | None = None,
) -> tuple[BaseModel, Path, Path | None, Path | None]:
    """Find and download everything the route trains from.

    Returns (base_spec, checkpoint_path, vae_path, text_encoder_path), with
    None for a file the route does not have.

    - ``custom``: a spec or a resolved BaseModel, in place of the route's list.
    - ``convrot_int8`` and ``config_overrides``: which files download
      (convrot_choice). The flags they need are on the returned base_spec.
    """
    if not route.t4_practical and "T4" in (environment.gpu_name() or ""):
        # Only a warning, before the large downloads.
        print(
            f"warning: {route.display_name} is not practical on a T4. It needs bf16, which a T4 does not "
            "support natively, and it is too large to train there at a usable speed. Pick an L4 or an A100 "
            "in Runtime > Change runtime type."
        )
    if isinstance(custom, BaseModel):
        base = custom
    else:
        base = custom_base(custom, route.key) if custom else route.base(base_key)
    choice = convrot_choice(route, convrot_int8, config_overrides)
    dit_file, dit_flags = _convrot_file(base, choice.dit, "convrot_int8")
    te_file, te_flags = None, ()
    if route.default_text_encoder:
        te_file, te_flags = _convrot_file(route.default_text_encoder, choice.text_encoder, "convrot_int8_text_encoder")
    if choice.note:
        print(" ".join([choice.note, *_convrot_effects(choice, dit_flags, te_flags)]))
    checkpoint = download(dit_file)
    vae = download(route.default_vae) if route.default_vae else None
    text_encoder = download(te_file) if te_file else None
    if dit_flags or te_flags:
        base = dataclasses.replace(base, convrot_flags=dit_flags + te_flags)
    return base, checkpoint, vae, text_encoder

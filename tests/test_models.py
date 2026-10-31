"""Custom-model resolution: prefix routing, Civitai API mapping, direct downloads."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
import requests

from kohya_trainer import models, registry

# --- prefix routing ---


def test_hf_repo_id_with_and_without_file():
    base = models.custom_base("user/repo")
    assert (base.repo_id, base.filename, base.download_url) == ("user/repo", None, None)
    base = models.custom_base("user/repo:model.safetensors")
    assert base.filename == "model.safetensors"


def test_hf_urls_parse_to_repo_and_file():
    base = models.custom_base("https://huggingface.co/user/repo")
    assert (base.repo_id, base.filename) == ("user/repo", None)
    base = models.custom_base("https://huggingface.co/user/repo/")
    assert (base.repo_id, base.filename) == ("user/repo", None)
    base = models.custom_base("https://huggingface.co/user/repo/blob/main/model.safetensors")
    assert (base.repo_id, base.filename) == ("user/repo", "model.safetensors")
    base = models.custom_base("https://huggingface.co/user/repo/resolve/main/sub/model.safetensors?download=true")
    assert (base.repo_id, base.filename) == ("user/repo", "sub/model.safetensors")
    assert base.download_url is None  # HF URLs go through hf_hub_download


def test_direct_safetensors_url():
    base = models.custom_base("https://example.com/files/model.safetensors?x=1")
    assert base.download_url == "https://example.com/files/model.safetensors?x=1"
    assert base.filename == "model.safetensors"


def test_direct_non_safetensors_url_rejected():
    with pytest.raises(ValueError, match="safetensors"):
        models.custom_base("https://example.com/model.ckpt")


def test_garbage_spec_rejected():
    with pytest.raises(ValueError, match="repo id"):
        models.custom_base("not-a-model")


def test_hf_dataset_url_is_rejected():
    with pytest.raises(ValueError, match="model repos"):
        models.custom_base("https://huggingface.co/datasets/squad")


def test_hf_url_mentioning_civitai_in_query_routes_to_hf():
    base = models.custom_base("https://huggingface.co/user/repo?ref=civitai.com")
    assert base.repo_id == "user/repo"


def test_civitai_image_link_is_rejected():
    with pytest.raises(ValueError, match="model page URL"):
        models.custom_base("https://civitai.com/images/12345")


# --- Civitai resolution (API mocked) ---

_FILE_V2 = {
    "name": "great_v2.safetensors",
    "type": "Model",
    "primary": True,
    "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "pruned"},
    "downloadUrl": "https://civitai.com/api/download/models/222",
}
_VERSION = {"id": 222, "name": "v2.0", "baseModel": "Illustrious", "description": "", "files": [_FILE_V2]}
_OLD_VERSION = {
    "id": 111,
    "name": "v1.0",
    "baseModel": "Illustrious",
    "description": "",
    "files": [
        {**_FILE_V2, "name": "great_v1.safetensors", "downloadUrl": "https://civitai.com/api/download/models/111"}
    ],
}
_MODEL = {"id": 99, "name": "Great Checkpoint", "type": "Checkpoint", "modelVersions": [_VERSION, _OLD_VERSION]}


@pytest.fixture
def civitai_api(monkeypatch):
    calls: list[str] = []

    def fake_json(url):
        calls.append(url)
        if "/api/v1/models/" in url:
            return _MODEL
        if "/api/v1/model-versions/" in url:
            return {**_VERSION, "model": {"name": "Great Checkpoint", "type": "Checkpoint"}}
        raise AssertionError(url)

    monkeypatch.setattr(models, "_civitai_json", fake_json)
    return calls


def test_civitai_model_url_picks_current_version(civitai_api):
    base = models.custom_base("https://civitai.com/models/99/great-checkpoint")
    assert base.filename == "great_v2.safetensors"
    assert base.download_url.endswith("/222")
    assert base.repo_id == "civitai:222"


def test_civitai_explicit_version_param_wins(civitai_api):
    base = models.custom_base("https://civitai.com/models/99?modelVersionId=111")
    assert base.filename == "great_v1.safetensors"


def test_civitai_download_url_resolves_its_version(civitai_api):
    base = models.custom_base("https://civitai.com/api/download/models/222")
    assert base.filename == "great_v2.safetensors"
    assert any("/model-versions/222" in call for call in civitai_api)


def test_host_matching_is_case_and_scheme_tolerant(civitai_api):
    assert models.custom_base("https://WWW.CIVITAI.COM/models/99").filename == "great_v2.safetensors"
    assert models.custom_base("civitai.com/models/99").filename == "great_v2.safetensors"
    assert models.custom_base("https://HUGGINGFACE.CO/user/repo").repo_id == "user/repo"


def test_civitai_non_checkpoint_is_rejected(monkeypatch):
    monkeypatch.setattr(models, "_civitai_json", lambda url: {**_MODEL, "type": "LORA"})
    with pytest.raises(ValueError, match="not a base checkpoint"):
        models.custom_base("https://civitai.com/models/99")


def test_resolve_network_weights_local_and_remote(tmp_path, monkeypatch):
    local = tmp_path / "my lora.safetensors"
    local.write_bytes(b"00")
    assert models.resolve_network_weights(str(local)) == local.resolve()
    downloads: list[str] = []
    monkeypatch.setattr(models, "download", lambda base: downloads.append(base.key) or tmp_path / "dl.safetensors")
    assert models.resolve_network_weights("user/repo:epoch-000008.safetensors") == tmp_path / "dl.safetensors"
    assert downloads == ["user/repo:epoch-000008.safetensors"]
    assert models.resolve_network_weights("user/repo") == tmp_path / "dl.safetensors"
    with pytest.raises(FileNotFoundError, match="network_weights"):
        models.resolve_network_weights(str(tmp_path / "nope.safetensors"))
    # A missing relative path with one slash must not go to the Hub as a repo id.
    with pytest.raises(FileNotFoundError, match="network_weights"):
        models.resolve_network_weights("output/lora.safetensors")
    with pytest.raises(ValueError, match="model page URL"):
        models.resolve_network_weights("https://civitai.com/images/12345")


def test_resolve_network_weights_accepts_civitai_lora(monkeypatch, tmp_path):
    """The continue-from field takes a LoRA or LoCon and refuses checkpoints."""
    monkeypatch.setattr(models, "_civitai_json", lambda url: {**_MODEL, "type": "LORA"})
    monkeypatch.setattr(models, "download", lambda base: tmp_path / "dl.safetensors")
    assert models.resolve_network_weights("https://civitai.com/models/99") == tmp_path / "dl.safetensors"
    monkeypatch.setattr(models, "_civitai_json", lambda url: _MODEL)  # type: Checkpoint
    with pytest.raises(ValueError, match="finished LoRA"):
        models.resolve_network_weights("https://civitai.com/models/99")


def test_civitai_lora_resolution_skips_the_vpred_hint(monkeypatch, capsys, tmp_path):
    vpred_version = {**_VERSION, "name": "V-Pred-1.0-Version"}
    monkeypatch.setattr(
        models, "_civitai_json", lambda url: {**_MODEL, "type": "LORA", "modelVersions": [vpred_version]}
    )
    monkeypatch.setattr(models, "download", lambda base: tmp_path / "dl.safetensors")
    models.resolve_network_weights("https://civitai.com/models/99")
    assert "v_parameterization" not in capsys.readouterr().out


def test_civitai_wrong_family_warns(civitai_api, capsys):
    models.custom_base("https://civitai.com/models/99", route_key="sd15")
    assert "tagged 'Illustrious'" in capsys.readouterr().out
    models.custom_base("https://civitai.com/models/99", route_key="illustrious")
    assert "tagged" not in capsys.readouterr().out


def test_civitai_vpred_hint_warns_with_the_flags(monkeypatch, capsys):
    vpred = {**_VERSION, "name": "V-Pred-1.0-Version"}
    monkeypatch.setattr(models, "_civitai_json", lambda url: {**_MODEL, "modelVersions": [vpred]})
    models.custom_base("https://civitai.com/models/99", route_key="sd15")
    out = capsys.readouterr().out
    assert "v_parameterization" in out and "debiased_estimation_loss" in out
    # The SDXL and Illustrious notebooks have a box for it.
    for route_key in ("sdxl", "illustrious"):
        models.custom_base("https://civitai.com/models/99", route_key=route_key)
        out = capsys.readouterr().out
        assert "looks like v-prediction" in out and "tick custom_base_is_vpred" in out
        assert "v_parameterization" not in out


# --- file-pick algorithm ---


def test_file_pick_scores_safetensor_fp16_pruned():
    files = [
        {"name": "old.ckpt", "type": "Model", "metadata": {"format": "PickleTensor"}},
        {
            "name": "fp32.safetensors",
            "type": "Model",
            "metadata": {"format": "SafeTensor", "fp": "fp32", "size": "full"},
        },
        {
            "name": "best.safetensors",
            "type": "Model",
            "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "pruned"},
        },
        {"name": "data.zip", "type": "Training Data", "metadata": {}},
    ]
    assert models._pick_civitai_file({"id": 1, "files": files}, "m")["name"] == "best.safetensors"


def test_file_pick_honors_primary_flag():
    files = [
        {
            "name": "best.safetensors",
            "type": "Model",
            "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "pruned"},
        },
        {
            "name": "site-pick.safetensors",
            "type": "Model",
            "primary": True,
            "metadata": {"format": "SafeTensor", "fp": "fp32", "size": "full"},
        },
    ]
    assert models._pick_civitai_file({"id": 1, "files": files}, "m")["name"] == "site-pick.safetensors"


def test_file_pick_rejects_pickle_only():
    files = [{"name": "old.ckpt", "type": "Model", "metadata": {"format": "PickleTensor"}}]
    with pytest.raises(ValueError, match="SafeTensor"):
        models._pick_civitai_file({"id": 1, "files": files}, "m")


def test_file_pick_requires_a_model_file():
    files = [{"name": "data.zip", "type": "Training Data", "metadata": {}}]
    with pytest.raises(ValueError, match="no model-weights"):
        models._pick_civitai_file({"id": 1, "files": files}, "m")


# --- direct download mechanics (requests mocked) ---


class _FakeResponse:
    def __init__(self, status=200, chunks=(b"",), json_body=None, headers=None):
        self.status_code = status
        self._chunks = chunks
        self._json = json_body or {}
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size):
        yield from self._chunks


class _FakeSession:
    responses: ClassVar[list[_FakeResponse]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        return _FakeSession.responses.pop(0)


def _url_base(
    repo_id: str = "civitai:222",
    filename: str = "m.safetensors",
    download_url: str = "https://civitai.com/api/download/models/222",
) -> registry.BaseModel:
    return registry.BaseModel(key="x", display_name="x", repo_id=repo_id, filename=filename, download_url=download_url)


@pytest.fixture
def fake_requests(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(models, "DOWNLOAD_CACHE", tmp_path / "cache")
    monkeypatch.setattr(requests, "Session", _FakeSession)
    _FakeSession.responses = []
    return _FakeSession


def test_download_url_streams_atomically_and_caches(fake_requests):
    fake_requests.responses = [_FakeResponse(307), _FakeResponse(200, chunks=(b"wei", b"ghts"))]
    path = models._download_url(_url_base())
    assert path.read_bytes() == b"weights"
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    fake_requests.responses = []  # cached hit must not touch the network
    assert models._download_url(_url_base()) == path


def test_download_url_shows_progress(fake_requests, capsys):
    # A large checkpoint takes minutes, so the download shows a progress bar.
    fake_requests.responses = [
        _FakeResponse(307),
        _FakeResponse(200, chunks=(b"wei", b"ghts"), headers={"Content-Length": "7"}),
    ]
    models._download_url(_url_base())
    assert "7.00/7.00" in capsys.readouterr().err


def test_hub_downloads_use_local_dir_real_files(tmp_path, monkeypatch):
    """The Hub cache gives a relative symlink to a blob, which the backend's readlink turns into
    '../../blobs/<sha>'. Downloads must be real files with their real names."""
    import huggingface_hub

    calls = {}

    def fake_download(repo_id, filename, local_dir=None):
        calls.update(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return str(tmp_path / "cache" / "user-repo" / filename)

    monkeypatch.setattr(models, "DOWNLOAD_CACHE", tmp_path / "cache")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    path = models.download(registry.BaseModel(key="k", display_name="k", repo_id="user/repo", filename="m.safetensors"))
    assert calls["local_dir"] == tmp_path / "cache" / "user-repo"
    assert path.name == "m.safetensors"


def test_download_url_401_names_the_civitai_token(fake_requests):
    fake_requests.responses = [_FakeResponse(401, json_body={"message": "requires you to be logged in"})]
    with pytest.raises(PermissionError, match="CIVITAI_TOKEN") as raised:
        models._download_url(_url_base())
    assert "requires you to be logged in" in str(raised.value)


@pytest.mark.parametrize(
    "body",
    [
        ValueError("not JSON"),
        ["a list"],
        {"message": 123},
        requests.exceptions.ChunkedEncodingError("the body was cut off"),
    ],
)
def test_download_url_401_without_a_message_still_names_the_token(fake_requests, body):
    fake_requests.responses = [_FakeResponse(401, json_body=body)]
    with pytest.raises(PermissionError, match="this download requires a Civitai login. Create"):
        models._download_url(_url_base())


def test_cache_namespaces_by_source_so_same_filenames_never_collide(fake_requests):
    fake_requests.responses = [_FakeResponse(307), _FakeResponse(200, chunks=(b"AAA",))]
    first = models._download_url(_url_base(repo_id="civitai:1"))
    fake_requests.responses = [_FakeResponse(307), _FakeResponse(200, chunks=(b"BBB",))]
    second = models._download_url(_url_base(repo_id="civitai:2"))
    assert first != second
    assert first.read_bytes() == b"AAA" and second.read_bytes() == b"BBB"


def test_unsafe_remote_filename_is_reduced_to_basename(fake_requests):
    fake_requests.responses = [_FakeResponse(307), _FakeResponse(200, chunks=(b"x",))]
    path = models._download_url(_url_base(filename="../../escape.safetensors"))
    assert path.name == "escape.safetensors"
    assert models.DOWNLOAD_CACHE in path.parents


def test_truncated_download_is_rejected_before_the_final_rename(fake_requests):
    fake_requests.responses = [
        _FakeResponse(307),
        _FakeResponse(200, chunks=(b"abc",), headers={"Content-Length": "10"}),
    ]
    with pytest.raises(OSError, match="truncated"):
        models._download_url(_url_base())
    assert not list(models.DOWNLOAD_CACHE.rglob("*.safetensors"))


def test_direct_links_skip_the_civitai_probe(fake_requests):
    # Exactly one response is queued: a Civitai first request uses it up.
    fake_requests.responses = [_FakeResponse(200, chunks=(b"w",))]
    path = models._download_url(_url_base(repo_id="", download_url="https://example.com/m.safetensors"))
    assert path.read_bytes() == b"w"


def test_download_file_streams_atomically_and_reuses_a_complete_file(fake_requests, tmp_path):
    fake_requests.responses = [_FakeResponse(200, chunks=(b"zip", b"data"), headers={"Content-Length": "7"})]
    target = tmp_path / "scratch" / "downloads" / "set.zip"
    assert models.download_file("https://example.com/set.zip", target) == target
    assert target.read_bytes() == b"zipdata"
    assert not target.with_suffix(".zip.tmp").exists()
    fake_requests.responses = []  # a second run must not touch the network
    assert models.download_file("https://example.com/set.zip", target) == target


# --- ConvRot INT8: one decision for the files and the flags ---

MOSS = registry.get("moss_image")


def _choice(value="auto", overrides=None, gpu=None, ram=float("inf")):
    return models.convrot_choice(MOSS, value, overrides, gpu_memory_gb=gpu, ram_gb=ram)


def test_convrot_on_off_and_overrides():
    assert (_choice("on").dit, _choice("on").text_encoder) == (True, True)
    assert (_choice("OFF").dit, _choice("OFF").text_encoder) == (False, False)
    assert _choice("on").note == "ConvRot INT8 is on, as convrot_int8 asks."
    # A True or False in config_overrides wins, model by model, and the note gives each reason.
    mixed = _choice("on", {"convrot_int8_text_encoder": False})
    assert (mixed.dit, mixed.text_encoder) == (True, False)
    assert mixed.note == (
        "ConvRot INT8 is on for the DiT (as convrot_int8 asks) and off for the text encoder (as config_overrides asks)."
    )
    t4 = _choice("auto", {"convrot_int8_text_encoder": False}, gpu=15.0)
    assert t4.note == (
        "ConvRot INT8 is on for the DiT (because this GPU has 15 GB of memory) and off for the text encoder "
        "(as config_overrides asks)."
    )
    both = _choice("auto", {"convrot_int8": True, "convrot_int8_text_encoder": True}, gpu=40.0, ram=80.0)
    assert both.note == "ConvRot INT8 is on, as config_overrides asks."
    only_dit = _choice("off", {"convrot_int8": True, "convrot_int8_text_encoder": None})
    assert (only_dit.dit, only_dit.text_encoder) == (True, False)
    # None removes the flag, so it means off for that model, even when the form says on.
    removed = _choice("on", {"convrot_int8": None})
    assert (removed.dit, removed.text_encoder) == (False, True)
    with pytest.raises(ValueError, match="takes True, False or None"):
        _choice("on", {"convrot_int8": "false"})
    with pytest.raises(ValueError, match="takes auto, on or off"):
        _choice("yes")
    # A route without ConvRot gets nothing, whatever is asked.
    assert models.convrot_choice(registry.get("anima"), "on") == models.ConvRotChoice()


@pytest.mark.parametrize(
    "gpu, ram, wanted, words",
    [
        (15.0, 53.0, True, "because this GPU has 15 GB of memory"),  # a T4
        (22.5, 53.0, False, "off, as this GPU has 22 GB and the machine has 53 GB of RAM, so"),  # an L4
        (40.0, 83.0, False, "the bf16 files are used, the reference quality"),  # an A100
        (40.0, 12.7, True, "because this machine has 13 GB of RAM"),
        (None, 12.7, True, "because this machine has 13 GB of RAM"),  # no GPU reading: RAM alone
        (None, 53.0, False, "off, as the machine has 53 GB of RAM (no GPU memory reading was found)"),
        (40.0, float("inf"), False, "off, as this GPU has 40 GB (the RAM could not be read)"),
        (None, float("inf"), False, "off, as neither the GPU memory nor the RAM could be read"),
    ],
)
def test_convrot_auto_goes_by_gpu_memory_and_ram(gpu, ram, wanted, words):
    choice = _choice("auto", gpu=gpu, ram=ram)
    assert (choice.dit, choice.text_encoder) == (wanted, wanted)
    assert words in choice.note
    assert choice.note.count("ConvRot INT8") == 1


def test_convrot_auto_reads_this_machine(monkeypatch):
    monkeypatch.setattr(models.environment, "gpu_memory_gb", lambda: 15.0)
    monkeypatch.setattr(models.environment, "total_ram_gb", lambda: 53.0)
    assert models.convrot_choice(MOSS, "auto").dit is True


def _fake_downloads(monkeypatch):
    fetched = []

    def download(base):
        fetched.append((base.repo_id, base.filename))
        return Path("/cache") / Path(base.filename or "model.safetensors").name

    monkeypatch.setattr(models, "download", download)
    return fetched


def test_prepare_base_downloads_the_int8_files_or_sets_the_flag(monkeypatch, capsys):
    fetched = _fake_downloads(monkeypatch)
    base, checkpoint, vae, text_encoder = models.prepare_base(MOSS, convrot_int8="on")
    assert checkpoint.name == "moss_image_base_int8_convrot.safetensors"
    assert text_encoder.name == "qwen3.5_9b_int8_convrot.safetensors"
    assert vae.name == "flux2-vae.safetensors"
    assert base.convrot_flags == ()  # the backend detects int8 files itself
    assert base.filename == MOSS.base().filename  # the base keeps its own name
    assert capsys.readouterr().out.count("ConvRot INT8 is on") == 1  # said once
    fetched.clear()
    base, checkpoint, _, text_encoder = models.prepare_base(MOSS, convrot_int8="off")
    assert (checkpoint.name, text_encoder.name) == ("moss_image_base_bf16.safetensors", "qwen3.5_9b_bf16.safetensors")
    assert base.convrot_flags == ()
    # A pasted base has no int8 file, so the backend quantizes it while it loads.
    fetched.clear()
    base, checkpoint, _, text_encoder = models.prepare_base(
        MOSS, custom="someone/moss-merge:merge.safetensors", convrot_int8="on"
    )
    assert fetched[0] == ("someone/moss-merge", "merge.safetensors")
    assert base.convrot_flags == ("convrot_int8",)
    assert text_encoder.name == "qwen3.5_9b_int8_convrot.safetensors"
    # The note says what each file does: the pasted base is not smaller.
    out = capsys.readouterr().out
    assert "The text encoder's int8 file is about half the size and GPU memory of the bf16 one." in out
    assert "The DiT downloads as given and turns into int8 while loading, if not int8 already" in out
    # With only the pasted base converted, nothing gets smaller to download.
    models.prepare_base(
        MOSS,
        custom="someone/moss-merge:merge.safetensors",
        convrot_int8="on",
        config_overrides={"convrot_int8_text_encoder": False},
    )
    out = capsys.readouterr().out
    assert "about half" not in out and "The DiT downloads as given" in out
    # The overrides the 1-click cell passes decide too.
    base, checkpoint, _, text_encoder = models.prepare_base(
        MOSS, convrot_int8="on", config_overrides={"convrot_int8": False}
    )
    assert (checkpoint.name, text_encoder.name) == (
        "moss_image_base_bf16.safetensors",
        "qwen3.5_9b_int8_convrot.safetensors",
    )


def test_a_text_encoder_without_an_int8_file_gets_its_flag(monkeypatch):
    import dataclasses

    _fake_downloads(monkeypatch)
    route = dataclasses.replace(
        MOSS, default_text_encoder=dataclasses.replace(MOSS.default_text_encoder, int8_filename=None)
    )
    base, _, _, text_encoder = models.prepare_base(route, convrot_int8="on")
    assert text_encoder.name == "qwen3.5_9b_bf16.safetensors"
    assert base.convrot_flags == ("convrot_int8_text_encoder",)


def test_prepare_base_leaves_other_routes_alone(monkeypatch):
    def no_reading():
        raise AssertionError("read the GPU on a route without ConvRot")

    monkeypatch.setattr(models.environment, "gpu_memory_gb", no_reading)
    fetched = _fake_downloads(monkeypatch)
    base, *_ = models.prepare_base(registry.get("anima"), convrot_int8="on")
    assert base.convrot_flags == ()
    assert fetched[0][1] == registry.get("anima").base().filename


def test_int8_files_sit_next_to_their_bf16_files():
    for route in registry.ROUTES.values():
        for model in (*route.base_models, route.default_text_encoder):
            if model is not None and model.int8_filename:
                assert route.convrot_int8, route.key
                assert model.int8_filename.endswith("_int8_convrot.safetensors"), model.key
                assert Path(model.int8_filename).parent == Path(model.filename).parent, model.key
                assert "w4a8" not in model.int8_filename


def test_prepare_base_takes_a_resolved_base_as_it_is(tmp_path, monkeypatch):
    """quick_train resolves a pasted link before the dataset work and hands the result on."""
    fake = tmp_path / "f.safetensors"
    fake.write_bytes(b"0")
    monkeypatch.setattr(models, "download", lambda spec: fake)
    monkeypatch.setattr(models, "custom_base", lambda *args, **kwargs: pytest.fail("the link was resolved twice"))
    resolved = models.BaseModel(
        key="https://civitai.com/models/99",
        display_name="great.safetensors",
        repo_id="civitai:222",
        filename="great.safetensors",
        download_url="https://civitai.com/api/download/models/222",
    )
    spec, checkpoint, _, _ = models.prepare_base(registry.get("illustrious"), custom=resolved)
    assert spec == resolved
    assert checkpoint == fake

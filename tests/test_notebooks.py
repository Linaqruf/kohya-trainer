"""Committed notebooks are build artifacts: they must match the emitter exactly."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import re
import textwrap
import types

import pytest

from conftest import ROOT, SD_SCRIPTS, needs_submodule


def _emitter():
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EMITTER = _emitter()
OUTPUTS = EMITTER.outputs()


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_committed_notebook_matches_emitter(path):
    assert path.is_file(), f"{path.name} is missing. Run python scripts/build_notebooks.py"
    assert path.read_text(encoding="utf-8") == OUTPUTS[path], (
        f"{path.name} differs from the emitter. Do not edit a notebook by hand. "
        "Edit scripts/build_notebooks.py and run it again."
    )


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_notebook_is_valid_and_marked_generated(path):
    nb = json.loads(OUTPUTS[path])
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    assert "build_notebooks.py" in first["source"]


def _route_of(name: str):
    from kohya_trainer import registry

    return registry.ROUTES[name.removesuffix(".ipynb").removesuffix("_advanced").removesuffix("_lora")]


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_notebook_asks_colab_for_its_route_gpu(path):
    nb = json.loads(OUTPUTS[path])
    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == _route_of(path.name).gpu


def test_gpu_note_points_to_a_paid_plan_and_names_bf16_only_where_it_trains():
    from kohya_trainer import registry

    for row in registry.ROUTES.values():
        note = EMITTER.gpu_note(row)
        assert f"an {row.gpu} GPU" in note
        assert "https://colab.research.google.com/signup" in note
        assert "Runtime > Change runtime type" in note
        for word in ("Colab Pro", "free", "Pro+", "AI Plus"):
            assert word not in note, (row.key, word)
        assert ("bf16" in note) == EMITTER._trains_bf16(row), row.key
        # A bf16 route says, before the run, what to change on a T4, or that
        # the model is not practical there.
        assert ("`train_batch_size` to 1" in note) == (EMITTER._trains_bf16(row) and row.t4_practical), row.key
        assert ("pick an L4 or an A100" in note) == (not row.t4_practical), row.key


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_cells_contain_no_logic(path):
    """The durability rule: cells call the package and implement nothing."""
    nb = json.loads(OUTPUTS[path])
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        assert "#@title" in source, "every code cell is a Colab form"
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("def ", "class ")), f"logic in a notebook cell: {stripped!r}"


def test_every_registry_row_has_both_flavors():
    from kohya_trainer import registry

    names = {p.name for p in OUTPUTS}
    for key in registry.ROUTES:
        assert f"{key}_lora.ipynb" in names
        assert f"{key}_lora_advanced.ipynb" in names


def _field_count(content: str) -> int:
    return content.count("#@param")


def _by_name(name: str) -> str:
    return next(content for path, content in OUTPUTS.items() if path.name == name)


def test_form_density_matches_the_hobbyist_audience():
    """People here change settings. A sparse form sends them elsewhere."""
    for path, content in OUTPUTS.items():
        if path.name.endswith("_advanced.ipynb"):
            assert _field_count(content) >= 40, f"{path.name}: {_field_count(content)} fields"
        else:
            assert _field_count(content) >= 18, f"{path.name}: {_field_count(content)} fields"


def test_placeholders_only_on_empty_string_fields():
    """Colab shows `placeholder` hint text only in a string field, while it is empty."""
    for path, content in OUTPUTS.items():
        nb = json.loads(content)
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            for line in cell["source"].splitlines():
                if "placeholder:" not in line:
                    continue
                assert re.search(r'=\s*""\s*#@param', line), f"{path.name}: placeholder on a non-empty default: {line}"
                assert 'type:"string"' in line, f"{path.name}: placeholder on a non-string field: {line}"
                # Colab's form parser turns a colon in a placeholder into invalid JSON.
                value = re.search(r'placeholder:"([^"]*)"', line).group(1)
                assert ":" not in value, f"{path.name}: colon breaks Colab's param parser: {line}"


def test_wandb_opt_in_present_in_both_flavors():
    for path, content in OUTPUTS.items():
        assert "log_to_wandb" in content, path.name


def test_route_conditional_sections():
    """Each route shows only the fields it uses: text encoder rate, flow, SNR and conv fields."""
    sd15 = _by_name("sd15_lora_advanced.ipynb")
    sdxl = _by_name("sdxl_lora_advanced.ipynb")
    anima = _by_name("anima_lora_advanced.ipynb")
    assert "text_encoder_lr" in sd15
    assert "text_encoder_lr" not in sdxl and "text_encoder_lr" not in anima
    assert "min_snr_gamma" in sd15 and "min_snr_gamma" in sdxl
    assert "min_snr_gamma" not in anima
    assert "noise_offset" in sd15 and "noise_offset" in sdxl
    assert "noise_offset" not in anima
    assert "timestep_sampling" in anima and "timestep_sampling" not in sdxl
    assert "conv_dim" in sdxl and "conv_dim" not in anima
    for path, content in OUTPUTS.items():
        assert "__" not in content, "unresolved placeholder leaked into a notebook"
        if path.name.endswith("_advanced.ipynb"):
            assert "gradient_accumulation_steps" in content, path.name
            assert "sample_negative_prompt" in content, path.name
            assert "network_weights" in content, path.name
            assert "tag_report" in content, path.name


# --- cells parse and call the package with real parameters ---

_SHELL_LINE = re.compile(r"^\s*[!%]")


def _parse_cell(source: str) -> ast.Module:
    # Colab form markers are plain comments. Shell lines (!cmd) and magics
    # (%cmd) are not Python, so they are blanked before parsing.
    lines = ["" if _SHELL_LINE.match(line) else line for line in source.splitlines()]
    return ast.parse("\n".join(lines))


def _code_cells(content: str) -> list[str]:
    return [cell["source"] for cell in json.loads(content)["cells"] if cell["cell_type"] == "code"]


def _in_package(obj) -> bool:
    return getattr(obj, "__module__", "").startswith("kohya_trainer")


def _forwarding_calls(func, var_kw: str) -> list[tuple[object, int, list[str], bool]]:
    """Calls inside ``func`` that pass its ``**var_kw`` on: (callee, positional
    count, explicit keywords, whether other star arguments are present)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(k.arg is None and isinstance(k.value, ast.Name) and k.value.id == var_kw for k in node.keywords):
            continue
        callee = None
        if isinstance(node.func, ast.Name):
            callee = func.__globals__.get(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            callee = getattr(func.__globals__.get(node.func.value.id), node.func.attr, None)
        starred = any(isinstance(a, ast.Starred) for a in node.args) or sum(k.arg is None for k in node.keywords) > 1
        found.append((callee, len(node.args), [k.arg for k in node.keywords if k.arg], starred))
    return found


def check_call(func, npos: int, keywords: list[str], partial: bool = False) -> list[str]:
    """Problems with calling ``func`` this way. Keywords that land in its
    ``**kwargs`` are followed into the call that receives them."""
    name = func.__qualname__
    # ast.parse accepts a repeated keyword. Only compiling the cell rejects it.
    repeated = sorted({k for k in keywords if keywords.count(k) > 1})
    if repeated:
        return [f"{name}: {repeated} passed twice"]
    sig = inspect.signature(func)
    params = sig.parameters.values()
    var_kw = next((p.name for p in params if p.kind is p.VAR_KEYWORD), None)
    named = [p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    own = [k for k in keywords if k in named] if var_kw else list(keywords)
    rest = [k for k in keywords if k not in own]
    bind = sig.bind_partial if partial else sig.bind
    try:
        bind(*[None] * npos, **dict.fromkeys(own))
    except TypeError as error:
        return [f"{name}: {error}"]
    if not rest:
        return []
    forwards = _forwarding_calls(func, var_kw)
    if not forwards:
        return [f"{name}: cannot follow **{var_kw} for {rest}"]
    problems = []
    for callee, callee_npos, explicit, starred in forwards:
        if callee is None:
            problems.append(f"{name}: cannot resolve where **{var_kw} goes")
            continue
        twice = sorted(set(rest) & set(explicit))
        if twice:
            problems.append(f"{name}: {twice} are also passed by {name} itself, so they arrive twice")
        merged = explicit + [k for k in rest if k not in explicit]
        problems += check_call(callee, callee_npos, merged, partial=partial or starred)
    return problems


def _resolve(node: ast.expr, names: dict):
    """The package object an expression names, or None when it names
    something else. Raises AttributeError when a package module lacks it."""
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, names)
        if isinstance(base, types.ModuleType):
            return getattr(base, node.attr)
    return None


def _import_names(tree: ast.Module, names: dict) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kohya_trainer"):
            for alias in node.names:
                try:
                    obj = importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    obj = getattr(importlib.import_module(node.module), alias.name)
                names[alias.asname or alias.name] = obj
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("kohya_trainer"):
                    top = alias.asname or alias.name.split(".")[0]
                    names[top] = importlib.import_module(alias.name if alias.asname else top)


def check_cells(sources: list[str]) -> tuple[list[str], set[str]]:
    """(problems, package calls checked) for one notebook's code cells.
    Imports carry from cell to cell, as they do in a running kernel."""
    names: dict = {}
    problems: list[str] = []
    checked: set[str] = set()
    for source in sources:
        tree = _parse_cell(source)
        _import_names(tree, names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            label = ast.unparse(node.func)
            try:
                func = _resolve(node.func, names)
            except AttributeError:
                problems.append(f"{label}: no such name in the package")
                continue
            if func is None or not callable(func) or not _in_package(func):
                continue
            checked.add(label)
            partial = any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords)
            keywords = [k.arg for k in node.keywords if k.arg]
            problems += [f"{label}(): {p}" for p in check_call(func, len(node.args), keywords, partial=partial)]
    return problems, checked


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_cells_parse_and_call_real_parameters(path):
    problems, checked = check_cells(_code_cells(OUTPUTS[path]))
    assert not problems, f"{path.name}: {problems}"
    expected = {"environment.install", "environment.load_tokens", "environment.mount_drive"}
    if path.name.endswith("_advanced.ipynb"):
        expected |= {"train.build_plan", "train.execute", "models.prepare_base", "tagging.tag_images"}
        expected |= {"dataset.prepare_dataset", "dataset.describe", "dataset.tag_report", "tagging.prune_into_trigger"}
    else:
        expected |= {"train.quick_train"}
    assert expected <= checked, f"{path.name}: checked only {sorted(checked)}"


def config_override_keys(sources: list[str], name: str = "config_overrides") -> tuple[set[str], list[str]]:
    """(keys, problems) for the dicts named ``name`` in the cells. A key that is not a plain string is a problem."""
    keys: set[str] = set()
    problems: list[str] = []
    for source in sources:
        for node in ast.walk(_parse_cell(source)):
            assigned = isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            )
            passed = isinstance(node, ast.keyword) and node.arg == name
            value = node.value if assigned or passed else None
            if value is None or isinstance(value, ast.Name):
                continue
            if not isinstance(value, ast.Dict):
                problems.append(f"{name} is not a dict literal: {ast.unparse(value)}")
                continue
            for key in value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
                else:
                    problems.append(f"{name} key is not a plain string: {ast.unparse(key) if key else '**'}")
    return keys, problems


@needs_submodule
@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_form_value_keys_are_backend_arguments(path):
    from kohya_trainer import config

    sources = _code_cells(OUTPUTS[path])
    keys, problems = config_override_keys(sources, "form_values")
    assert not problems, f"{path.name}: {problems}"
    assert {"learning_rate", "max_train_epochs"} <= keys, f"{path.name}: {sorted(keys)}"
    if path.name.endswith("_advanced.ipynb"):
        assert {"max_train_steps", "save_last_n_epochs"} <= keys, f"{path.name}: {sorted(keys)}"
    override_keys, problems = config_override_keys(sources)
    assert not problems, f"{path.name}: {problems}"
    config.validate_keys(dict.fromkeys(keys | override_keys), SD_SCRIPTS, _route_of(path.name).entrypoint)


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_override_dicts_start_empty_and_reach_the_plan(path):
    """The two dicts start empty in both notebooks, under one comment line, and reach the plan."""
    sources = _code_cells(OUTPUTS[path])
    source = next(s for s in sources if "config_overrides = {}" in s)
    assert "dataset_overrides = {}" in source
    lines = source.splitlines()
    above = lines[lines.index("config_overrides = {}") - 1]
    assert above == EMITTER.DICTS_COMMENT
    assert "config_overrides=config_overrides" in source and "dataset_overrides=dataset_overrides" in source
    assert "form_values=" in source
    for name in ("config_overrides", "dataset_overrides"):
        keys, problems = config_override_keys(sources, name)
        assert keys == set() and not problems, (path.name, name, keys, problems)


@needs_submodule
def test_config_override_check_flags_a_key_the_backend_lacks():
    from kohya_trainer import config

    sources = _code_cells(_by_name("sdxl_lora_advanced.ipynb"))
    assert any('"max_train_steps"' in source for source in sources)
    broken = [source.replace('"max_train_steps"', '"max_train_stepz"') for source in sources]
    keys, problems = config_override_keys(broken, "form_values")
    assert not problems and "max_train_stepz" in keys
    with pytest.raises(ValueError, match="max_train_stepz"):
        config.validate_keys(dict.fromkeys(keys), SD_SCRIPTS, _route_of("sdxl_lora_advanced.ipynb").entrypoint)


def test_config_override_check_flags_a_key_it_cannot_read():
    cell = '#@title Bad\nextra = {}\nconfig_overrides = {"seed": 1, **extra}\n'
    keys, problems = config_override_keys([cell])
    assert keys == {"seed"} and problems


def test_cells_have_no_shell_lines_today():
    # If this fails, the parse check above blanks real lines.
    for path, content in OUTPUTS.items():
        for source in _code_cells(content):
            assert not any(_SHELL_LINE.match(line) for line in source.splitlines()), path.name


_GOOD_CELL = """#@title Train
from kohya_trainer import tagging, train
plan = train.quick_train(model="sdxl", dataset="d", project_name="p", network_dim=8, resume=True)
tagging.tag_images("d", repo_id="r", append_tags=True)
"""


def test_call_check_passes_a_correct_cell():
    problems, checked = check_cells([_GOOD_CELL])
    assert problems == []
    assert checked == {"train.quick_train", "tagging.tag_images"}


@pytest.mark.parametrize(
    "line, expected",
    [
        ('train.quick_train(model="sdxl", dataset="d", project_name="p", netwrok_dim=8)', "netwrok_dim"),
        ('train.quick_train(model="sdxl", dataset="d", project_name="p", checkpoint_path="c")', "twice"),
        ('train.quick_train(model="sdxl", dataset="d", project_name="p", resume=1, resume=2)', "twice"),
        ('train.quick_train(dataset="d", project_name="p")', "model"),
        ('tagging.tag_images("d", repo_idd="r")', "repo_idd"),
        ('train.quick_trian(model="sdxl")', "no such name"),
        ("environment.install(strict=True)", "strict"),
    ],
)
def test_call_check_flags_a_bad_call(line, expected):
    cell = "#@title Broken\nfrom kohya_trainer import environment, tagging, train\n" + line + "\n"
    problems, _ = check_cells([cell])
    assert problems and expected in " ".join(problems), problems


# --- the README links every route's notebooks ---

_BADGE = re.compile(
    r"\[!\[Open in Colab\]\(https://colab\.research\.google\.com/assets/colab-badge\.svg\)\]"
    r"\(https://colab\.research\.google\.com/github/[^)\s]+/notebooks/([a-z0-9_-]+)\.ipynb\)"
)


def readme_routes(text: str) -> list[tuple[str, str, list[str], str]]:
    """(label, route key, notebook names, GPU) for each row of the notebook
    tables. A row belongs to the route its notebook links name."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("| Model |")]
    assert headers, "the README has no notebook table"
    rows = []
    for header in headers:
        for line in lines[header + 2 :]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            links = [_BADGE.fullmatch(cell) for cell in cells[2:]]
            names = [link.group(1) if link else cell for link, cell in zip(links, cells[2:])]
            key = names[0].removesuffix("_lora") if names and names[0].endswith("_lora") else ""
            rows.append((cells[0], key, names, cells[1] if len(cells) > 1 else ""))
    return rows


def test_readme_has_one_row_per_route():
    from kohya_trainer import registry

    rows = readme_routes((ROOT / "README.md").read_text(encoding="utf-8"))
    keys = [key for _, key, _, _ in rows]
    assert sorted(keys) == sorted(registry.ROUTES), rows
    for label, key, names, gpu in rows:
        assert names == [f"{key}_lora", f"{key}_lora_advanced"], f"row {label!r}: {names}"
        assert label == registry.ROUTES[key].display_name, f"row {label!r} links {key} notebooks"
        assert gpu == registry.get(key).gpu, f"row {label!r}: GPU {gpu!r}"


def test_readme_check_flags_a_missing_or_wrong_row():
    from kohya_trainer import registry

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    first = next(line for line in text.splitlines() if "sd15_lora.ipynb" in line)
    dropped = readme_routes(text.replace(first + "\n", ""))
    assert sorted(key for _, key, _, _ in dropped) != sorted(registry.ROUTES)
    swapped = readme_routes(text.replace(first, first.replace("sd15_lora_advanced", "sdxl_lora_advanced")))
    assert any(names != [f"{key}_lora", f"{key}_lora_advanced"] for _, key, names, _ in swapped)
    wrong_gpu = readme_routes(text.replace(first, first.replace("| L4 |", "| T4 |")))
    assert any(gpu != registry.get(key).gpu for _, key, _, gpu in wrong_gpu)


def test_readme_names_no_plan_and_links_the_paid_plans_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://colab.research.google.com/signup" in text
    for word in ("Colab Pro", "Free-tier", "free T4", "Pro+", "AI Plus"):
        assert word not in text, word


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_footer_says_to_stop_a_session_left_running(path):
    footer = json.loads(OUTPUTS[path])["cells"][-1]
    assert footer["cell_type"] == "markdown"
    assert "Runtime > Manage sessions" in footer["source"]
    assert "compute units" in footer["source"]
    assert "12 h" not in footer["source"]


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_footer_says_a_shared_outfit_can_bring_the_subject_in_without_the_trigger(path):
    footer = json.loads(OUTPUTS[path])["cells"][-1]["source"]
    assert footer.count("without the trigger word") == 1
    assert "one outfit" in footer and "lower the LoRA weight" in footer


def test_vpred_hint_shows_only_where_a_vpred_base_can_train():
    from kohya_trainer import registry

    for row in registry.ROUTES.values():
        path = next(p for p in OUTPUTS if p.name == f"{row.key}_lora_advanced.ipynb")
        source = json.loads(OUTPUTS[path])["cells"][3]["source"]  # the Base model cell
        assert source.startswith("#@title Base model"), row.key
        if EMITTER._is_flow(row):  # a flow-matching DiT has no v-pred bases
            assert "v-pred" not in source
            continue
        assert "`sample_sampler`" in source and '"euler"' in source, row.key
        listed = any(base.v_prediction for base in row.base_models)
        assert ("The v-pred model in the list sets them for you." in source) == listed, row.key


@pytest.mark.parametrize(
    "path", sorted(p for p in OUTPUTS if not p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name
)
def test_oneclick_checks_the_project_name_before_mounting_drive(path):
    train_cell = json.loads(OUTPUTS[path])["cells"][2]["source"]
    assert train_cell.index("train.check_project_name(project_name)") < train_cell.index("environment.mount_drive()")


def _advanced_cell(content: str, title: str) -> str:
    cells = json.loads(content)["cells"]
    return next(c["source"] for c in cells if c["cell_type"] == "code" and c["source"].startswith(f"#@title {title}"))


def test_tagger_dropdown_defaults_to_eva02_and_takes_a_typed_repo():
    for path, content in OUTPUTS.items():
        if not path.name.endswith("_advanced.ipynb"):
            continue
        cell = _advanced_cell(content, "Captions")
        line = next(line for line in cell.splitlines() if line.startswith("tagger ="))
        assert '"SmilingWolf/wd-eva02-large-tagger-v3"' in line.split("#@param")[0]
        assert "swinv2" not in line
        assert line.endswith("{allow-input: true}"), line


def test_dataset_cell_names_every_source_and_the_numbered_folders():
    for path, content in OUTPUTS.items():
        if not path.name.endswith("_advanced.ipynb"):
            continue
        cell = _advanced_cell(content, "Dataset")
        for form in (
            "a .zip",
            "link to a .zip",
            "Hugging Face dataset repo of image files or one .zip (user/name)",
            "10_mychar",
        ):
            assert form in cell, f"{path.name}: {form}"
        assert "dataset.describe(dataset_dir)" in cell


def test_captions_cell_edits_tags_and_offers_rating_tags_only_off_anima():
    for path, content in OUTPUTS.items():
        if not path.name.endswith("_advanced.ipynb"):
            continue
        cell = _advanced_cell(content, "Captions")
        for name in ("character_tags_first", "remove_tags"):
            assert f"\n{name} = " in cell and f"{name}=" not in cell.split("tagging.tag_images")[0], name
        assert "tagging.remove_tags(" in cell
        assert cell.index("tagging.remove_tags(") < cell.index("dataset.tag_report(")
        assert 'caption_extension = ".txt"' in cell and "caption_extension=caption_extension" in cell
        assert 'backup_dir=workspace / "captions-backup"' in cell
        rates = "use_rating_tags" in cell
        from kohya_trainer import tagging

        assert rates == (_route_of(path.name).entrypoint in tagging.RATING_TAG_ENTRYPOINTS), path.name
        assert not (rates and path.name.startswith(("anima", "moss_image"))), path.name


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_every_notebook_offers_to_fold_hair_and_eye_tags_into_the_trigger(path):
    sources = _code_cells(OUTPUTS[path])
    cell = next(source for source in sources if "\nprune_appearance_tags = " in source)
    lines = cell.splitlines()
    box = next(i for i, line in enumerate(lines) if line.startswith("prune_appearance_tags = "))
    assert lines[box] == 'prune_appearance_tags = True #@param {type:"boolean"}'
    assert lines[box - 1].startswith("#@markdown After the tagger runs") and "trigger word" in lines[box - 1]
    assert lines[box - 2].startswith("trigger_word = ")  # the box sits under the trigger word
    if path.name.endswith("_advanced.ipynb"):
        # Only after this cell's own tagging, with a trigger word, never on
        # captions append_tags kept, and before remove_tags and the tag report.
        tagged = cell.index("    tagging.prune_into_trigger(dataset_dir, caption_extension=caption_extension)")
        assert cell.index("if run_tagger:") < cell.index("tagging.tag_images(") < tagged
        assert "    if prune_appearance_tags and trigger_word.strip() and not append_tags:\n" in cell
        assert tagged < cell.index("tagging.remove_tags(") < cell.index("dataset.tag_report(")
    else:
        assert "    prune_appearance_tags=prune_appearance_tags,\n" in cell
        assert "It works only with a trigger word." in lines[box - 1]


@pytest.mark.parametrize("path", sorted(p for p in OUTPUTS if p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name)
@pytest.mark.parametrize("trigger, prunes", [("zkx", True), ("   ", False), ("", False)])
def test_advanced_captions_cell_prunes_only_with_a_trigger_word(path, trigger, prunes, tmp_path, monkeypatch):
    from kohya_trainer import dataset, tagging

    cell = next(source for source in _code_cells(OUTPUTS[path]) if "\nprune_appearance_tags = " in source)
    field = next(line for line in cell.splitlines() if line.startswith("trigger_word = "))
    # The tagger is ticked here: pruning follows only this cell's own tagging,
    # and MOSS-Image starts with it unticked.
    tagger_box = next(line for line in cell.splitlines() if line.startswith("run_tagger = "))
    pruned = []
    monkeypatch.setattr(tagging, "tag_images", lambda *args, **kwargs: None)
    monkeypatch.setattr(tagging, "prune_into_trigger", lambda *args, **kwargs: pruned.append(args))
    monkeypatch.setattr(dataset, "tag_report", lambda *args, **kwargs: "")
    # The cell is code this repository writes, run with its calls replaced.
    namespace = {"dataset_dir": tmp_path, "workspace": tmp_path}
    exec(cell.replace(field, f"trigger_word = {trigger!r}").replace(tagger_box, "run_tagger = True"), namespace)  # noqa: S102
    assert pruned == ([(tmp_path,)] if prunes else [])


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_the_tagger_box_starts_at_the_routes_default(path):
    """MOSS-Image starts with the tagger off, so a set with no captions trains on the trigger word."""
    route = _route_of(path.name)
    name = "run_tagger" if path.name.endswith("_advanced.ipynb") else "auto_tag"
    box = next(
        line for source in _code_cells(OUTPUTS[path]) for line in source.splitlines() if line.startswith(f"{name} = ")
    )
    assert box == f'{name} = {route.default_auto_tag} #@param {{type:"boolean"}}'
    assert route.default_auto_tag == (route.key != "moss_image")
    if not path.name.endswith("_advanced.ipynb"):
        title = json.loads(OUTPUTS[path])["cells"][0]["source"]
        expected = (
            "each one trains on the trigger word alone" if route.key == "moss_image" else "the WD tagger tags them"
        )
        assert expected in title


def _dropdown(content: str, name: str) -> str | None:
    """The #@param line of the field ``name``, or None when the notebook has no such field."""
    for source in _code_cells(content):
        for line in source.splitlines():
            if line.startswith(f"{name} = ") and "#@param" in line:
                return line
    return None


@pytest.mark.parametrize("path", sorted(p for p in OUTPUTS if p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name)
def test_optimizer_and_scheduler_dropdowns_take_a_typed_name(path):
    from kohya_trainer import train

    for name, offered, accepted in (
        ("optimizer", list(train.OPTIMIZERS), train.OPTIMIZERS),
        ("lr_scheduler", EMITTER.SCHEDULERS, train.SCHEDULERS),
    ):
        line = _dropdown(OUTPUTS[path], name)
        assert line and line.endswith("{allow-input: true}"), (path.name, line)
        listed = json.loads(line.split("#@param", 1)[1].split("{allow-input")[0])
        assert listed == offered and set(listed) <= set(accepted), (path.name, listed)
    assert "Lion8bit" not in _dropdown(OUTPUTS[path], "optimizer")


@pytest.mark.parametrize("path", sorted(OUTPUTS), ids=lambda p: p.name)
def test_vpred_box_sits_under_custom_base_on_the_sdxl_family(path):
    row = _route_of(path.name)
    sources = _code_cells(OUTPUTS[path])
    has_box = any("custom_base_is_vpred = False #@param" in source for source in sources)
    assert has_box == (row.key in ("sdxl", "illustrious")) == row.custom_vpred_option, path.name
    if has_box:
        cell = next(source for source in sources if "custom_base_is_vpred = False" in source)
        lines = cell.splitlines()
        box = next(i for i, line in enumerate(lines) if line.startswith("custom_base_is_vpred = "))
        assert lines[box - 2].startswith("custom_base = "), path.name
        assert "does nothing without a custom base" in lines[box - 1]
        assert any("custom_base_is_vpred=custom_base_is_vpred" in source for source in sources), path.name


_ONECLICK_SECTIONS = {
    "Project": ["project_name", "mount_google_drive", "resume_from_state"],
    "Model": ["base_model", "custom_base", "custom_base_is_vpred"],
    "Dataset": ["dataset", "auto_tag", "trigger_word", "prune_appearance_tags", "resolution", "num_repeats"],
    "Training": [
        "max_train_epochs",
        "max_train_steps",
        "train_batch_size",
        "optimizer",
        "learning_rate",
        "text_encoder_lr",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_scheduler_num_cycles",
        "min_snr_gamma",
    ],
    "Network": ["network_type", "network_dim", "network_alpha"],
    "Output": ["save_every_n_epochs", "sample_prompt", "upload_to_hf", "log_to_wandb"],
}


def _oneclick_fields(path) -> dict[str, list[str]]:
    """The 1-click Train cell's fields, by section, in order."""
    source = json.loads(OUTPUTS[path])["cells"][2]["source"]
    sections: dict[str, list[str]] = {}
    current = ""
    for line in source.splitlines():
        header = re.match(r"#@markdown \*\*(\w+)\*\*", line)
        if header:
            current = header.group(1)
            sections[current] = []
        elif "#@param" in line:
            sections[current].append(line.split(" = ", 1)[0])
    return sections


@pytest.mark.parametrize(
    "path", sorted(p for p in OUTPUTS if not p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name
)
def test_oneclick_form_has_the_training_knobs_in_order(path):
    row = _route_of(path.name)
    expected = {name: list(fields) for name, fields in _ONECLICK_SECTIONS.items()}
    if not row.custom_vpred_option:
        expected["Model"].remove("custom_base_is_vpred")
    if row.convrot_int8:
        expected["Model"].append("convrot_int8")
    if EMITTER._caches_te(row):
        expected["Training"].remove("text_encoder_lr")
    if EMITTER._is_flow(row):
        expected["Training"].remove("min_snr_gamma")
    assert _oneclick_fields(path) == expected
    count = {"sd15": 28, "sdxl": 28, "illustrious": 28, "anima": 26, "moss_image": 27}[row.key]
    assert _field_count(OUTPUTS[path]) == count, path.name


@pytest.mark.parametrize(
    "path", sorted(p for p in OUTPUTS if not p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name
)
def test_oneclick_defaults_come_from_the_preset(path):
    from kohya_trainer import config

    row = _route_of(path.name)
    preset = config.load_preset(row.preset)
    source = json.loads(OUTPUTS[path])["cells"][2]["source"]
    values = {}
    for line in source.splitlines():
        if "#@param" in line:
            name, _, rest = line.partition(" = ")
            values[name] = ast.literal_eval(rest.split(" #@param", 1)[0])
    assert values["learning_rate"] == preset["learning_rate"]
    assert values["max_train_epochs"] == preset["max_train_epochs"]
    assert values["lr_scheduler"] == preset["lr_scheduler"]
    assert values["lr_warmup_steps"] == preset.get("lr_warmup_steps", 0.0)
    assert values["save_every_n_epochs"] == preset["save_every_n_epochs"]
    assert values["optimizer"] == preset["optimizer_type"]
    if "min_snr_gamma" in values:
        assert values["min_snr_gamma"] == preset["min_snr_gamma"]
    # Only SD1.5 says its learning rate, restarts and run length go together.
    assert ("about 1,500 steps" in source) == (row.key == "sd15"), path.name


@pytest.mark.parametrize(
    "path", sorted(p for p in OUTPUTS if not p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name
)
def test_oneclick_intro_says_every_setting_has_a_default(path):
    intro = json.loads(OUTPUTS[path])["cells"][0]["source"]
    assert "Every setting has a default" in intro
    assert "`config_overrides`" in intro
    assert "one cell at a time, with the tagger settings and tag editing" in intro
    assert "the scheduler, the optimizer or any sd-scripts setting, use the" not in intro


def test_multires_noise_is_no_longer_a_field_but_still_a_key():
    for path, content in OUTPUTS.items():
        assert _dropdown(content, "multires_noise") is None, path.name
        for source in _code_cells(content):
            assert "multires_noise_iterations" not in source, path.name
    # The keys still work through config_overrides, and the footer says how.
    footer = json.loads(_by_name("sdxl_lora_advanced.ipynb"))["cells"][-1]["source"]
    assert '"multires_noise_iterations": 6, "multires_noise_discount": 0.3' in footer


def test_noise_offset_help_and_the_illustrious_note_follow_current_recipes():
    from kohya_trainer import registry

    for name in ("sd15_lora_advanced.ipynb", "sdxl_lora_advanced.ipynb", "illustrious_lora_advanced.ipynb"):
        content = _by_name(name)
        assert _dropdown(content, "noise_offset").startswith("noise_offset = 0 #@param"), name
        assert "Most current recipes leave `noise_offset` at 0" in content, name
        assert "0.03 is the usual value" not in content, name
    note = registry.get("illustrious").tuning_note
    assert "0.03" not in note and "multires_noise" not in note
    assert "Leave `noise_offset` at 0" in note


@needs_submodule
@pytest.mark.parametrize("path", sorted(p for p in OUTPUTS if p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name)
def test_footer_extra_settings_are_backend_arguments_where_shown(path):
    from kohya_trainer import config

    row = _route_of(path.name)
    footer = json.loads(OUTPUTS[path])["cells"][-1]["source"]
    shown = EMITTER.extra_settings(row)
    keys = {key for entry_keys, _, _ in shown for key in entry_keys}
    config.validate_keys(dict.fromkeys(keys), SD_SCRIPTS, row.entrypoint)
    for entry_keys, where, _ in shown:
        assert where in ("config_overrides", "dataset_overrides")
        for key in entry_keys:
            assert f'"{key}"' in footer, (path.name, key)
    assert ("clip_skip" in keys) == (row.key == "sd15")
    assert ("min_timestep" in keys) == ("multires_noise_iterations" in keys) == (not EMITTER._is_flow(row))
    for left_out in ("ip_noise_gamma", "huber", "loraplus", "scale_weight_norms", "lr_scheduler_power"):
        assert left_out not in footer, (path.name, left_out)
    # Caption dropout works with the text cache on Anima and MOSS-Image only.
    if row.entrypoint in config.TE_CACHE_KEEPS_DROPOUT:
        assert "It works with the cached text-encoder outputs." in footer
    elif EMITTER._caches_te(row):
        assert "It needs text-encoder caching turned off" in footer


@pytest.mark.parametrize("path", sorted(p for p in OUTPUTS if p.name.endswith("_advanced.ipynb")), ids=lambda p: p.name)
def test_footer_says_how_the_presets_the_form_and_the_dicts_stack(path):
    footer = json.loads(OUTPUTS[path])["cells"][-1]["source"]
    text = " ".join(footer.split())
    assert "give the form its defaults and fill in every setting the form does not show" in text
    assert "The form wins over the preset, and the two dicts win over both." in text
    assert "To change a value the form shows, change the field." in text
    assert "keep the entries already there" not in text


PHILOSOPHY = """kohya-trainer trains LoRAs on Colab the way LoRA trainers already work: sd-scripts names for every setting, the usual folder and caption habits, and the steps people take to get from images to a finished LoRA.
- The 1-click notebook is one press: add your images and train. Every setting has a default. It is not a cut-down trainer: its cell takes any sd-scripts setting, so a power user can use it too.
- The advanced notebook runs the same steps one cell at a time, so you can check and redo each stage.
- Nothing goes in unless it helps you finish a LoRA in one sitting."""


@pytest.mark.parametrize("name", ["README.md", "CONTRIBUTING.md", "scripts/build_notebooks.py"])
def test_the_philosophy_is_written_down_word_for_word(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    if name.endswith(".py"):
        text = EMITTER.__doc__
    # Quote marks, list indents and line breaks differ between the files. The words do not.
    flat = " ".join(line.strip().removeprefix(">").strip() for line in text.splitlines())
    assert " ".join(PHILOSOPHY.split()) in " ".join(flat.split()), name


def _footer(content: str) -> str:
    return json.loads(content)["cells"][-1]["source"]


def test_footers_promise_a_file_per_epoch_only_with_the_default_save_interval():
    """save_every_n_epochs is a form field, and 0 saves only the final file."""
    for path, content in OUTPUTS.items():
        footer = " ".join(_footer(content).split())
        for promise in ("per epoch", "every finished epoch is already saved"):
            for match in re.finditer(re.escape(promise), footer, re.IGNORECASE):
                before = footer[max(0, match.start() - 90) : match.start()]
                assert "`save_every_n_epochs` at 1 (the default)" in before or "save_every_n_epochs = 1" in before, (
                    path.name,
                    promise,
                )
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "with `save_every_n_epochs` at 1 (the default) a file for each epoch" in readme


def test_optimizer_help_names_package_optimizers_in_plain_words():
    assert "module.Class" not in EMITTER.OPTIMIZER_HELP
    assert "one from an installed package as package.Name, such as pytorch_optimizer.CAME" in (EMITTER.OPTIMIZER_HELP)
    sentences = [s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", EMITTER.OPTIMIZER_HELP.removeprefix("#@markdown "))]
    assert max(len(s.split()) for s in sentences) <= 20, sentences


def test_advanced_footer_says_folder_names_win_over_the_dataset_overrides():
    for path, content in OUTPUTS.items():
        if path.name.endswith("_advanced.ipynb"):
            footer = " ".join(_footer(content).split())
            assert "own number and name (such as 10_mychar) win over num_repeats and class_tokens" in footer, path.name


def test_flow_fields_take_their_defaults_and_choices_from_the_preset(monkeypatch):
    """The flow fields show the preset's values, so the form never overrides them."""
    from kohya_trainer import config, registry

    anima = registry.get("anima")
    fields = EMITTER.flow_fields(anima)
    assert 'timestep_sampling = "sigmoid" #@param ["sigmoid", "shift"]' in fields
    assert "discrete_flow_shift = 1.0 #@param" in fields and "sigmoid_scale = 1.3 #@param" in fields
    assert '`sigmoid_scale` works with "sigmoid" or "shift"' in fields
    assert "adjusts the default sampling" not in fields
    assert "Try 1.3" not in fields  # already the default

    real = config.load_preset

    def preset_with(**flow):
        return lambda name: {**real(name), **flow}

    monkeypatch.setattr(EMITTER.config, "load_preset", preset_with(sigmoid_scale=1.0))
    assert "Try 1.3 for sharper fine detail." in EMITTER.flow_fields(anima)
    monkeypatch.setattr(EMITTER.config, "load_preset", preset_with(timestep_sampling="shift", discrete_flow_shift=3.0))
    fields = EMITTER.flow_fields(anima)
    assert 'timestep_sampling = "shift" #@param ["sigmoid", "shift"]' in fields
    assert "discrete_flow_shift = 3.0 #@param" in fields
    assert "Try 3.0" not in fields  # the advice for a sigmoid default

    monkeypatch.setattr(EMITTER.config, "load_preset", preset_with(timestep_sampling="sigma"))
    fields = EMITTER.flow_fields(anima)
    assert 'timestep_sampling = "sigma" #@param ["sigmoid", "shift", "sigma"]' in fields
    # sigma reads the shift but not the scale (library/flux_train_utils.py).
    assert '`discrete_flow_shift` works only when `timestep_sampling` is "shift" or "sigma"' in fields
    assert '`sigmoid_scale` works with "sigmoid" or "shift".' in fields


@needs_submodule
def test_flow_help_matches_which_samplers_read_each_setting():
    """The help's sampler lists come from the backend's timestep sampling."""
    path = SD_SCRIPTS / "library" / "flux_train_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    [func] = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_noisy_model_input_and_timesteps"
    ]
    readers = {"discrete_flow_shift": set(), "sigmoid_scale": set()}

    def samplers_in(test) -> set[str]:
        tests_sampler = any(isinstance(n, ast.Attribute) and n.attr == "timestep_sampling" for n in ast.walk(test))
        return {n.value for n in ast.walk(test) if isinstance(n, ast.Constant)} if tests_sampler else set()

    def visit(node, current: set[str]) -> None:
        if isinstance(node, ast.If) and samplers_in(node.test):
            for child in node.body:
                visit(child, samplers_in(node.test))
            for child in node.orelse:
                visit(child, current)
            return
        if isinstance(node, ast.Attribute) and node.attr in readers:
            readers[node.attr] |= current
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    for statement in func.body:
        visit(statement, set())
    # "sigma" and "truncated_logit_normal" read the shift through the shifted scheduler, which the module names.
    shift_aware = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_SHIFT_AWARE_TIMESTEP_SAMPLING"
    ]
    readers["discrete_flow_shift"] |= {c.value for c in ast.walk(shift_aware[0].value) if isinstance(c, ast.Constant)}
    assert readers["discrete_flow_shift"] == set(EMITTER.FLOW_SHIFT_SAMPLERS)
    assert readers["sigmoid_scale"] == set(EMITTER.FLOW_SCALE_SAMPLERS)


@needs_submodule
def test_moss_image_samplers_are_known_to_the_backend():
    """The preset's sampler and those the tips name are choices of the pinned MOSS-Image trainer.

    sd-scripts does not check choices for values from the TOML config, and an unknown
    sampler falls back to uniform sampling without a word.
    """
    from kohya_trainer import config, registry

    moss = registry.get("moss_image")
    tree = ast.parse((SD_SCRIPTS / "library" / "moss_image_train_utils.py").read_text(encoding="utf-8"))
    [call] = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and n.args and getattr(n.args[0], "value", None) == "--timestep_sampling"
    ]
    [choices] = [k.value for k in call.keywords if k.arg == "choices"]
    known = {c.value for c in choices.elts}
    # Every mention of the sampler in the tips uses the config_overrides spelling, so none goes unchecked.
    named = re.findall(r'"timestep_sampling": "([^"]+)"', moss.tuning_note)
    assert moss.tuning_note.count("timestep_sampling") == len(named)
    named = set(named) | {config.load_preset(moss.preset)["timestep_sampling"]}
    assert "truncated_logit_normal" in named
    assert named <= known, named - known
    keys = set(re.findall(r'"(\w+)": ', moss.tuning_note))
    assert {"timestep_sampling", "logit_mean", "logit_std", "discrete_flow_shift"} <= keys
    config.validate_keys(dict.fromkeys(keys), SD_SCRIPTS, moss.entrypoint)


@needs_submodule
def test_moss_image_default_schedule_is_upstream_recipe():
    """The preset and the tip give the pre-training schedule that the pinned trainer's help names.

    The sign of logit_mean decides which end of the noise range most steps train at.
    """
    from kohya_trainer import config, registry

    moss = registry.get("moss_image")
    source = (SD_SCRIPTS / "library" / "moss_image_train_utils.py").read_text(encoding="utf-8")
    found = re.search(
        r"--timestep_sampling (\w+) --logit_mean (\S+) --logit_std (\S+) --discrete_flow_shift ([\d.]+)", source
    )
    assert found, "the trainer's help no longer names the pre-training schedule"
    sampler, mean, std, shift = found.groups()
    preset = config.load_preset(moss.preset)
    assert preset["timestep_sampling"] == sampler == "truncated_logit_normal"
    assert (preset["logit_mean"], preset["logit_std"], preset["discrete_flow_shift"]) == (
        float(mean),
        float(std),
        float(shift),
    )
    assert float(mean) < 0
    assert (
        f'"timestep_sampling": "{sampler}", "logit_mean": {mean}, "logit_std": {std}, "discrete_flow_shift": {shift}'
        in moss.tuning_note
    )
    # The cut and the uniform share are set too, at the backend's defaults, which are the pre-training values.
    tree = ast.parse((SD_SCRIPTS / "library" / "args.py").read_text(encoding="utf-8"))
    defaults = {
        n.args[0].value: next(k.value.value for k in n.keywords if k.arg == "default")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and n.args
        and getattr(n.args[0], "value", None) in ("--timestep_cutoff", "--timestep_uniform_ratio")
    }
    assert (preset["timestep_cutoff"], preset["timestep_uniform_ratio"]) == (0.95, 0.01)
    assert defaults == {"--timestep_cutoff": 0.95, "--timestep_uniform_ratio": 0.01}


def test_blocks_to_swap_shows_only_where_the_trainer_swaps_blocks():
    for path, content in OUTPUTS.items():
        route = _route_of(path.name)
        field = _dropdown(content, "blocks_to_swap")
        if route.max_blocks_to_swap and path.name.endswith("_advanced.ipynb"):
            assert field == 'blocks_to_swap = 0 #@param {type:"integer"}', path.name
            assert f"The most is {route.max_blocks_to_swap}." in content, path.name
            assert '"blocks_to_swap": blocks_to_swap' in "\n".join(_code_cells(content)), path.name
        else:
            assert field is None and "blocks_to_swap" not in content, path.name


def test_moss_image_notebooks_carry_the_route_defaults_and_notes():
    from kohya_trainer import registry

    moss = registry.get("moss_image")
    for name in ("moss_image_lora.ipynb", "moss_image_lora_advanced.ipynb"):
        content = _by_name(name)
        cells = json.loads(content)["cells"]
        assert moss.about_note in cells[0]["source"], name  # under the title
        # The tagger starts off on this route. The note says how to turn it on.
        assert "tick `auto_tag` (`run_tagger` in the advanced notebook)" in moss.about_note
        if name.endswith("_advanced.ipynb"):
            assert _dropdown(content, "run_tagger") == 'run_tagger = False #@param {type:"boolean"}'
        else:
            assert _dropdown(content, "auto_tag") == 'auto_tag = False #@param {type:"boolean"}'
        assert _dropdown(content, "train_batch_size") == 'train_batch_size = 1 #@param {type:"integer"}', name
        # Seven epochs: later epochs bring the training outfit into other prompts.
        assert _dropdown(content, "max_train_epochs") == 'max_train_epochs = 7 #@param {type:"integer"}', name
        # Base is the default training base, and SFT can be picked too.
        assert _dropdown(content, "base_model") == (
            'base_model = "moss-image-base" #@param ["moss-image-base", "moss-image-sft"]'
        ), name
        assert "The defaults set alpha equal to dim" in content, name
        assert "alpha to half of dim" not in content, name
        # The LoHa and LoKr help gives no T4 batch hint where a T4 is not practical.
        assert "LoHa and LoKr ignore these two fields" in content, name
        assert "On a T4, if it runs out of memory, set `train_batch_size` to 1" not in content, name
    for name in ("anima_lora.ipynb", "sdxl_lora_advanced.ipynb"):
        content = _by_name(name)
        assert _dropdown(content, "train_batch_size") == 'train_batch_size = 2 #@param {type:"integer"}', name
        # Anima trains at alpha equal to rank, as its author does.
        halved = name.startswith("sdxl")
        assert ("The defaults set alpha to half of dim" in content) == halved, name
        assert ("The defaults set alpha equal to dim" in content) == (not halved), name
        assert "On a T4, if it runs out of memory, set `train_batch_size` to 1" in content, name
    oneclick = _by_name("moss_image_lora.ipynb")
    assert moss.training_note in oneclick
    advanced = _by_name("moss_image_lora_advanced.ipynb")
    assert _dropdown(advanced, "timestep_sampling") == (
        'timestep_sampling = "truncated_logit_normal" #@param ["sigmoid", "shift", "truncated_logit_normal"]'
    )
    assert _dropdown(advanced, "discrete_flow_shift").startswith("discrete_flow_shift = 1.0 #@param")
    # The help says which shift each offered sampler needs.
    assert 'Keep 1.0 for "truncated_logit_normal". For "shift", use 3.0.' in "\n".join(_code_cells(advanced))
    footer = json.loads(advanced)["cells"][-1]["source"]
    assert "`caption_dropout_rate` is 0.1 by default for this model" in footer
    assert "It works with the cached text-encoder outputs." in footer
    assert "train_text_embedder=True" in footer and "---===---" in footer
    for word in ("plan", "review", "PR", "session ledger"):
        assert f" {word} " not in moss.tuning_note + moss.about_note, word


def test_convrot_choice_sits_where_the_models_download():
    """The field is in the cell that downloads, so the files and the flags come from one choice."""
    for path, content in OUTPUTS.items():
        route = _route_of(path.name)
        field = _dropdown(content, "convrot_int8")
        if not route.convrot_int8:
            assert field is None and "convrot_int8" not in content, path.name
            continue
        assert field == 'convrot_int8 = "auto" #@param ["auto", "on", "off"]', path.name
        if path.name.endswith("_advanced.ipynb"):
            cell = _advanced_cell(content, "Base model")
            assert "convrot_int8=convrot_int8)" in cell
            configure = _advanced_cell(content, "Configure")
            assert "convrot_int8" not in configure
            assert "run this cell again" in cell
        else:
            assert "    convrot_int8=convrot_int8," in "\n".join(_code_cells(content))
        assert "about half the GPU memory, RAM and download" in content
        assert "the reference quality" in content


def test_lokr_rate_hint_only_on_sdxl_routes():
    # The LoKr rate hint is for the SDXL family. The other routes have no such hint.
    hint = "If you pick lokr, set `learning_rate` to about 1e-3"
    for path, content in OUTPUTS.items():
        sdxl_family = path.name.startswith(("sdxl_", "illustrious_"))
        assert (hint in content) == sdxl_family, path.name

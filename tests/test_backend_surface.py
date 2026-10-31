"""What we rely on in the pinned backend source: the flags we pass and how it caches captions."""

from __future__ import annotations

import ast
import inspect
import warnings

from kohya_trainer import config, dataset, registry, tagging

from conftest import SD_SCRIPTS, needs_submodule

TAGGER_SCRIPT = SD_SCRIPTS / "finetune" / "tag_images_by_wd14_tagger.py"


def parse_source(script) -> ast.Module:
    # Some upstream help strings use invalid escapes, which warn on parse.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
        return ast.parse(script.read_text(encoding="utf-8-sig"))


def parser_options(script, function: str = "setup_parser") -> set[str]:
    """Every option string given to add_argument inside ``function``."""
    tree = parse_source(script)
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    assert found, f"{function}() not found in {script.name}"
    options = set()
    for node in ast.walk(found[0]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    options.add(arg.value)
    return options


def every_tag_flag_on() -> list[str]:
    """build_tag_command with each optional flag switched on."""
    switches = {}
    for name, param in inspect.signature(tagging.build_tag_command).parameters.items():
        if param.kind is not param.KEYWORD_ONLY or name == "sd_scripts_dir":
            continue
        if isinstance(param.default, bool):
            switches[name] = True
        elif param.default == "":
            switches[name] = "some_tag"
    assert {"undesired_tags", "remove_underscore", "append_tags", "recursive"} <= set(switches)
    return tagging.build_tag_command("images", sd_scripts_dir=SD_SCRIPTS, **switches)


@needs_submodule
def test_tagger_flags_exist_in_the_pinned_script():
    options = parser_options(TAGGER_SCRIPT)
    assert "--onnx" in options and "--repo_id" in options
    command = every_tag_flag_on()
    assert command[1] == str(TAGGER_SCRIPT)
    flags = [part.partition("=")[0] for part in command[2:] if part.startswith("--")]
    assert len(flags) >= 10
    unknown = sorted(set(flags) - options)
    assert not unknown, f"flags missing from {TAGGER_SCRIPT.name} at the pin: {unknown}"


# --- what the backend does with captions when it caches text-encoder outputs ---


def functions_by_name(tree: ast.Module) -> dict[str, list[ast.FunctionDef]]:
    found: dict[str, list[ast.FunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, []).append(node)
    return found


def called_names(node: ast.AST) -> set[str]:
    """Names of the functions and methods called inside ``node``."""
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name):
                names.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                names.add(call.func.attr)
    return names


def reaches(defs: dict[str, list[ast.FunctionDef]], start: str, target: str) -> bool:
    """Whether ``start`` calls ``target``, directly or through other functions in the same file."""
    seen, todo = set(), [start]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for func in defs.get(name, []):
            calls = called_names(func)
            if target in calls:
                return True
            todo.extend(calls & set(defs))
    return False


@needs_submodule
def test_text_encoder_cache_gets_caption_prefix_and_suffix():
    """The cached caption carries caption_prefix, where the trigger word goes, and caption_suffix."""
    defs = functions_by_name(parse_source(SD_SCRIPTS / "library" / "dataset.py"))

    adders = defs.get("add_caption_prefix_suffix", [])
    assert adders, "add_caption_prefix_suffix() not found in library/dataset.py"
    read = {node.attr for node in ast.walk(adders[0]) if isinstance(node, ast.Attribute)}
    assert {"caption_prefix", "caption_suffix"} <= read

    feeders = [f for f in defs.get("new_cache_text_encoder_outputs", []) if "cache_batch_outputs" in called_names(f)]
    assert len(feeders) == 1, "the method that fills the text-encoder cache moved"
    feeder = feeders[0]
    batch_calls = [
        call
        for call in ast.walk(feeder)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "cache_batch_outputs"
    ]
    assert batch_calls and all(len(call.args) == 5 for call in batch_calls), "captions no longer handed to the cache"
    assert reaches(defs, feeder.name, "add_caption_prefix_suffix"), "cache captions skip caption_prefix/suffix"

    # Our routes' cache strategies encode the captions they are handed, not the stored ones.
    for name in ("strategy_sdxl.py", "strategy_anima.py", "strategy_moss_image.py"):
        methods = functions_by_name(parse_source(SD_SCRIPTS / "library" / name)).get("cache_batch_outputs", [])
        assert len(methods) == 1, f"cache_batch_outputs() not found once in {name}"
        method = methods[0]
        assert "captions" in [arg.arg for arg in method.args.args], name
        raw = [node for node in ast.walk(method) if isinstance(node, ast.Attribute) and node.attr == "caption"]
        assert not raw, f"{name} encodes the raw caption stored on each image"


@needs_submodule
def test_disk_text_encoder_cache_is_reused_without_a_caption_check():
    """For SDXL and Anima, the backend reuses a disk text cache without a caption check, so
    refresh_text_caches removes old files. If upstream adds a check, that step can go."""
    for name in ("strategy_sdxl.py", "strategy_anima.py"):
        checks = functions_by_name(parse_source(SD_SCRIPTS / "library" / name)).get(
            "is_disk_cached_outputs_expected", []
        )
        assert len(checks) == 1, f"is_disk_cached_outputs_expected() not found once in {name}"
        params = [arg.arg for arg in checks[0].args.args]
        assert "captions" not in params, f"{name} now checks captions when reusing a cache: {params}"


def _caching_strategy_class(entrypoint: str) -> tuple[str, str] | None:
    """(strategy module, class) the trainer builds its text-encoder cache with."""
    methods = functions_by_name(parse_source(SD_SCRIPTS / entrypoint)).get(
        "get_text_encoder_outputs_caching_strategy", []
    )
    for method in methods:
        for call in ast.walk(method):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.attr.endswith("CachingStrategy")
            ):
                return call.func.value.id, call.func.attr
    return None


@needs_submodule
def test_text_cache_suffixes_match_the_pinned_strategies():
    """Old text caches are found by file-name suffix, so a suffix renamed upstream must fail here."""
    entrypoints = {route.entrypoint for route in registry.ROUTES.values()}
    assert set(dataset.TEXT_CACHE_SUFFIXES) <= entrypoints
    for entrypoint in sorted(entrypoints):
        found = _caching_strategy_class(entrypoint)
        if found is None:
            assert entrypoint not in dataset.TEXT_CACHE_SUFFIXES, entrypoint
            continue
        module, name = found
        tree = parse_source(SD_SCRIPTS / "library" / f"{module}.py")
        [cls] = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]
        suffixes = {
            target.id: node.value.value
            for node in cls.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.endswith("_NPZ_SUFFIX")
        }
        assert len(suffixes) == 1, f"{name} has no single cache suffix: {suffixes}"
        [(constant, value)] = suffixes.items()
        assert dataset.TEXT_CACHE_SUFFIXES.get(entrypoint) == value, entrypoint
        [path_method] = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "get_outputs_npz_path"]
        used = {node.attr for node in ast.walk(path_method) if isinstance(node, ast.Attribute)}
        assert constant in used and "splitext" in used, f"{name} names its cache files another way"


@needs_submodule
def test_text_cache_args_are_real_arguments():
    for entrypoint, names in dataset.TEXT_CACHE_ARGS.items():
        assert set(names) <= config.known_args(SD_SCRIPTS, entrypoint), entrypoint
    # Anima's cache stores each image's caption dropout rate.
    source = (SD_SCRIPTS / "library" / "strategy_anima.py").read_text(encoding="utf-8")
    assert "caption_dropout_rate=caption_dropout_rate" in source


def _trainers_passing(keyword: str) -> set[str]:
    """The trainers, of our routes and of the sets below, that ask the dataset
    whether text caching is allowed with ``keyword=True``."""
    entrypoints = {route.entrypoint for route in registry.ROUTES.values()}
    entrypoints |= config.TE_CACHE_KEEPS_DROPOUT | config.TE_CACHE_KEEPS_CANDIDATES
    found = set()
    for entrypoint in sorted(entrypoints):
        for call in ast.walk(parse_source(SD_SCRIPTS / entrypoint)):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "is_text_encoder_output_cacheable"
                and any(
                    kw.arg == keyword and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in call.keywords
                )
            ):
                found.add(entrypoint)
    return found


@needs_submodule
def test_caption_dropout_with_text_caching_matches_the_trainers():
    """check_compat lets caption_dropout_rate through only where the trainer
    asks the dataset with cache_supports_dropout=True."""
    assert _trainers_passing("cache_supports_dropout") == set(config.TE_CACHE_KEEPS_DROPOUT)


@needs_submodule
def test_caption_candidates_with_text_caching_match_the_trainers():
    """Several candidates per caption pass with text caching only where the trainer's cache keeps
    them all. The scan uses the backend's default separator."""
    assert _trainers_passing("supports_multiple_caption_candidates") == set(config.TE_CACHE_KEEPS_CANDIDATES)
    subset = (SD_SCRIPTS / "library" / "subset.py").read_text(encoding="utf-8")
    assert f'DEFAULT_MULTILINE_CAPTION_SEPARATOR = "{dataset.DEFAULT_MULTILINE_CAPTION_SEPARATOR}"' in subset

# Contributing to Kohya Trainer

This guide owns setup, the rules, the checks, and the steps that need the
maintainer. [AGENTS](AGENTS.md) adds the rules for coding agents. The human
maintainer owns the project and its direction.

## What the project is for

A feature that does not fit this stays out, however easy it is to add.

> kohya-trainer trains LoRAs on Colab the way LoRA trainers already work: sd-scripts names for every setting, the usual folder and caption habits, and the steps people take to get from images to a finished LoRA.
> - The 1-click notebook is one press: add your images and train. Every setting has a default. It is not a cut-down trainer: its cell takes any sd-scripts setting, so a power user can use it too.
> - The advanced notebook runs the same steps one cell at a time, so you can check and redo each stage.
> - Nothing goes in unless it helps you finish a LoRA in one sitting.

The code, the tests and the pinned backend show what runs today. The
maintainer's current intent says what it must do. Check a backend claim against
the pinned `external/sd-scripts`, not against memory or upstream docs.

## Setting up

You need Python 3.10 or newer. CI runs 3.10, 3.12 and 3.13 (the Colab version).

```bash
git clone --recurse-submodules <repository-url> kohya-trainer
cd kohya-trainer
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"      # Windows: .venv\Scripts\python.exe
```

The package does not need torch. Only the training machine installs the full
stack (`kohya_trainer.environment.install()`). The checks, as CI runs them:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python scripts/build_notebooks.py && git add -N notebooks && git diff --exit-code notebooks/
```

`pytest -rs` lists why each test was skipped. The backend tests need
`external/sd-scripts` checked out, and CI fails without it.

## Layout

All logic lives in the package. Notebook cells only read the form and call it.

- `src/kohya_trainer/`
  - `train.py`: re-exports the names the notebooks call.
  - `plan.py`: `build_plan`, its checks before downloads, `execute` and `quick_train`.
  - `settings.py`: the `build_plan` steps that fill a run's settings.
  - `optimizers.py`: optimizer and scheduler names, checks and defaults.
  - `network.py`: network types, sizes and `network_args`.
  - `dataset_checks.py`: dataset arguments, caption checks and the step estimate.
  - `previews.py`: the sample preview size, prompt and prompt file.
  - `resume.py`: finding a saved state and the checks before a resume.
  - `dataset.py`: dataset preparation, the dataset-config TOML, tag counts and the text-cache refresh.
  - `tagging.py`: the WD tagger from the submodule, and tag editing.
  - `models.py`: base model and `network_weights` downloads, and the ConvRot choice.
  - `environment.py`: everything specific to Colab and the machine.
  - `config.py`: presets to a flat config, checked against the backend's arguments.
  - `registry.py`: one row per model route.
  - `pins.py`: the backend commit and the install rules.
  - `presets/*.toml`: training defaults as commented data.
- `scripts/build_notebooks.py` builds both notebooks for every route.
  `scripts/canary_smoke.py` is the weekly check of the Colab setup.
- `notebooks/*.ipynb`: generated. Never edit them by hand.
- `external/sd-scripts`: the backend, pinned to one commit. Read it for its
  contract. Never copy its code into ours.
- `.github/workflows/`: `ci.yml` checks every pull request and every push to
  the default branch.
  `canary.yml` runs the canary weekly on a CPU-only runner: it installs the
  stack, imports every training script, builds a dry-run plan for every model
  and checks the Hub files. It opens an issue when it breaks. In a fork or another
  copy, a second job checks the public default branch, which the notebooks clone.

## Rules

- **No logic in notebook cells.** A test enforces it.
- **Two notebooks per route from one source.** Settings stack in this order:
  preset, form, recommended optimizer settings, then `config_overrides` and
  `dataset_overrides`, which always win.
- **A new model is one registry row**, plus a preset if its defaults differ.
- **Every emitted key is a real backend argument.** The backend ignores unknown
  keys without a warning, so `validate_keys` checks each one at the pin.
- **A text cache is rebuilt when its input changes.** The backend reuses cached
  text-encoder outputs without checking the text encoder. Before launch,
  `execute` compares a record of the captions, trigger word and text encoder,
  and removes the old text-cache files if it differs. Latent caches stay.
  Planning never deletes files.
- **Defaults, not gates.** A hardware rule is a default the user can override.
  Refuse only what the backend rejects or crashes on, at plan time.
- **A backend bump is one reviewed change**: the submodule,
  `pins.SD_SCRIPTS_COMMIT`, and a check of the arguments and every route's plan.
- **Secrets never enter a file**, a command line or a log. A test scans the tree
  for token-shaped strings.

When you change X, also update Y:

- A registry row or the builder: rebuild the notebooks.
- A preset: `tests/test_config.py` and the argument check at the pin.
- A form field: both templates in `scripts/build_notebooks.py`, then rebuild. Update the README if it
  describes the field.
- The pin, `environment.py` or the setup cell: run the canary.

## Making a change

- Branch from an up-to-date default branch. Never commit on it directly.
- Stage explicit paths. Never use `git add -A`, `git add .` or `git add -u`.
- Make one pull request per feature, with Conventional Commits, squash merged.
- Read each command's own exit status.
- If a failure is unexpected, reproduce it on the default branch first. Never
  hide it with a skip or a weaker assertion.
- Test through the code you change, with fakes that keep the real contract. A
  regression test must fail when its fix is removed.

## Evidence levels

Report the level a claim reached: (1) tests and lint, (2) the canary, (3) a
real Colab run from setup to samples and resume, (4) the LoRA file loading
where people generate images. Only a real Colab run proves the user path.

## Privacy and permission

Nothing committed contains a credential, an account name, a host address or a
local absolute path. Users' datasets and LoRA files never go into tests or docs.

These steps need the maintainer, or a delegation he has given:

- Merging a pull request or pushing to a shared branch.
- Publishing: the public repository, a release, a public upload.
- Spending money: paid Colab compute, paid APIs or services.
- Deleting something that has no copy elsewhere.

A delegation covers only what he named. A green build is evidence, not
permission.

## How we write

Write for a reader with none of the history, in simple technical English. A
comment states a constraint the code does not show. Comments never point to a
plan, review, session or pull request. A refused input gives the reason and the
way out. Use the LoRA community's words. Describe what the notebooks do now,
not the project's past.

# Agent instructions

These rules apply to every coding agent in this repository. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first. All of it applies to agents.
[CLAUDE.md](CLAUDE.md) points Claude Code here. Local agent folders (`.agents/`,
`.codex/`, `.claude/`) are ignored by git.

## Who decides

The human maintainer owns the project. Agents are contributors. Weigh any
contributor's reasoning, your own too, against the code, the tests, the pinned
backend and the maintainer's current intent. Explain a disagreement by what it
changes for the person who runs a notebook. Knowledge the project needs goes in
the repository, not only in agent memory.

## Before you change code

1. Check the branch and the uncommitted changes. Keep unrelated work as it is.
2. Read the code, its callers and its tests. For an argument, schema or
   entrypoint, read the pinned backend source too.
3. The registry, `build_plan`, the presets, `environment.py`, `pins.py`, the
   notebook builder and secrets handling are sensitive. Say what a change there
   trades off.
4. Name the outcome and the checks that will show it.

## Build and verify

Make the smallest coherent change. Apply new conventions only to code you touch.
Size the checks to the change:

- Backend contract, silent run behavior, secrets, the pin: an independent check
  of the plan, adversarial review, and a check of each fix by someone else.
- Ordinary features: the full checks, plus one reviewer when a shared seam
  changes.
- Mechanical and docs changes: the deterministic checks.

Report the command, its exit status, its skips, what stayed unchecked and the
evidence level. Do not edit a checkout while its suite runs. A builder's or
reviewer's result is evidence, never permission to merge, spend or publish. The
orchestrator's own run decides a merge. Turn a repeated defect into a
deterministic check.

Delegate only concrete, independent work. Each task names its model, effort,
files and design, and carries the privacy list for its machine.

## Files, processes and private state

- Read a file before you replace or delete it.
- Leave private material unread, and say what you could not establish without it.
- Work in your own clone. Delegated agents never touch the maintainer's
  checkout. The orchestrating session starts there, and can fast-forward the
  default branch only while there are no uncommitted changes.
- Real Colab runs happen in the maintainer's account. Agents prepare the steps
  and read the results.
- Never attach to or stop a process you did not start. Stop your own by its
  exact process id.
- Never print, log or commit a secret. Never read an employer's repository.

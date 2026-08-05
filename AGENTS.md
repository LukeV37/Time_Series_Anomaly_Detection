# AGENTS.md

## Scope

This repo currently has two real codepaths with different assumptions:
- `src/atlas/pbeast_fetcher/` plus `scripts/atlas/fetch_one_run.py` for ATLAS PBeast fetch and time-series alignment.
- `src/preprocessing/` for YAML-driven NumPy preprocessing pipelines.

`docs/atlas/README.md` and `docs/spt/README.md` are the top-level docs entrypoints for the domain codepaths. There is no root `README`, no `pyproject.toml`, no `pytest.ini`, and no CI workflow in the repo root, so do not guess standard commands that are not defined here.

## Environment

Use `setup.sh` first. It sources `export.sh`, creates or activates the venv, upgrades `pip`, and installs `requirements.txt`.
- Default venv is `<repo>/venv`.
- If `VENV_DIR` is set, `setup.sh` uses `$VENV_DIR/time_series_anomaly_detection` instead.

`export.sh` defines repo-local defaults for shared storage paths. Agents should read it before changing any path handling because the code relies on these env vars:
- `OUTPUT_DIR`
- `PBEAST_VENV_DIR`
- `VENV_DIR`
- `ATLAS_DATA_DIR`
- `PBEAST_HTML_DIR`
- `SPT_DATA_DIR`

Important ATLAS fetcher quirk:
- `src/atlas/pbeast_fetcher/pbeast_fetcher.py` requires `PBEAST_HTML_DIR` when `html_path` is not passed explicitly.
- `scripts/atlas/fetch_one_run.py` is more forgiving: it defaults `--html-dir` to `$PBEAST_HTML_DIR` if set, otherwise to bundled package data under `src/atlas/pbeast_fetcher/data`.

## Entry Points

ATLAS fetch flow:
- Run `python scripts/atlas/fetch_one_run.py --help` for the real CLI.
- It injects `src/` into `sys.path` itself, so running the script directly from repo root is expected.
- Verified defaults:
  - config: `src/atlas/pbeast_fetcher/configs/config.yaml`
  - sources: `src/atlas/pbeast_fetcher/configs/sources.yaml`
  - output: `<repo>/output`
  - merge strategy: `s2`
- `--merge-strategy baseline` exists but is explicitly described in code as slow and intended for verification.

Preprocessing flow:
- Core runner is `src/preprocessing/pipeline.py`.
- YAML loading in `src/utils/config_loader.py` resolves relative config paths from package-relative search roots, so calls like `load_config("configs/hlt_pipeline.yaml")` remain cwd-independent.
- Built-in pipeline configs live in `src/preprocessing/configs/`.

SPT script status:
- `scripts/spt/infer_spt_v2.py` imports `anldq.*`, but no `anldq` package exists in this repository and `requirements.txt` does not declare it.
- Treat SPT scripts as depending on code or environments outside this repo unless you verify otherwise before editing.

## Verification

There is no verified repo-wide test command in this checkout.

Use focused checks instead of inventing a full validation pipeline:
- `bash setup.sh`
- `python scripts/atlas/fetch_one_run.py --help`
- For preprocessing-only changes, a small import smoke test from repo root is safer than guessing a test suite, for example loading a known YAML config and constructing `PreprocessingPipeline`.

Do not claim lint, typecheck, or test coverage unless you actually find and run the corresponding config-driven command.

## Review Focus

When asked to code review this repo, prioritize issues around:
- hidden external dependencies and path assumptions (`export.sh`, shared storage env vars, `anldq` imports)
- scripts that mutate `sys.path` instead of using installed packages
- mismatches between docs and executable defaults, especially around HTML/config/source locations
- absence of automated tests/CI for the codepaths being changed

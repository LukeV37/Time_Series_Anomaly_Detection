# AGENTS.md

## Scope

This repo has three real local codepaths with different assumptions:
- `src/atlas/pbeast_fetcher/` plus `scripts/atlas/fetch/fetch_one_run.py` for ATLAS PBeast fetch and time-series alignment.
- `src/preprocessing/` for YAML-driven NumPy preprocessing pipelines.
- `src/training/` plus `scripts/spt/train_tranad.py` for the minimal local SPT TranAD training path.

There is no root `README`, no `pyproject.toml`, no `pytest.ini`, and no CI workflow in the repo root. Prefer the script CLIs and source files over guessed project-wide commands.

## Environment

Run `bash setup.sh` first. It sources `export.sh`, creates or activates the venv, upgrades `pip`, and installs `requirements.txt`.
- Default venv: `<repo>/venv`
- If `VENV_DIR` is set: `$VENV_DIR/time_series_anomaly_detection`

Read `export.sh` before changing path handling. Verified env vars used by local code/docs:
- `OUTPUT_DIR`
- `PBEAST_VENV_DIR`
- `VENV_DIR`
- `ATLAS_DATA_DIR`
- `PBEAST_HTML_DIR`
- `SPT_DATA_DIR`
- `SPT_DATA_DIR_BENCHMARK`

## Entry Points

ATLAS fetch:
- Real CLI: `python scripts/atlas/fetch/fetch_one_run.py --help`
- The script injects `src/` into `sys.path`; run it from the repo root.
- Verified defaults:
  - config: `src/atlas/pbeast_fetcher/configs/config.yaml`
  - sources: `src/atlas/pbeast_fetcher/configs/sources.yaml`
  - output: `<repo>/output`
  - html dir: `$PBEAST_HTML_DIR` if set, otherwise bundled data under `src/atlas/pbeast_fetcher/data`
  - merge strategy: `s2`
- `--merge-strategy baseline` is present but described in code as slow and intended for verification.
- Lower-level quirk: `src/atlas/pbeast_fetcher/pbeast_fetcher.py` itself requires `PBEAST_HTML_DIR` when `html_path` is omitted.

ATLAS preprocessing:
- Real CLI: `python scripts/atlas/preprocessing/run_atlas_preprocessing.py 520705`
- Defaults to config `src/preprocessing/configs/atlas_pipeline.yaml`.
- If `--csv-path` is not passed, the loader resolves from `--data-root` or `$ATLAS_DATA_DIR` plus the run number.

Shared preprocessing:
- Core runner: `src/preprocessing/pipeline.py`
- Current pipeline contract is stateless array processing: loaders return `(data, metadata)`, and registered steps operate on NumPy arrays, optionally receiving `metadata` when their signature supports it.
- `src/utils/config_loader.py` resolves relative config paths from cwd first, then `src/preprocessing/` and `src/training/`. Calls like `PreprocessingPipeline.from_config_file("configs/atlas_pipeline.yaml")` are intentionally cwd-tolerant.
- Built-in pipeline configs live in `src/preprocessing/configs/`.
- `src/preprocessing/configs/spt_pipeline.yaml` currently references `save_labels`, `split`, and `robust_scale`, but those steps are not registered in `src/preprocessing/registry.py`. Treat that config and `scripts/spt/run_preprocessing.py` as stale until the SPT preprocessing path is redesigned.

SPT local path vs legacy path:
- Local training CLI: `python scripts/spt/train_tranad.py --config src/training/configs/spt_tranad.yaml`
- Local training is a minimal standalone path in `src/training/`; it loads a preprocessing `.npz`, builds sliding windows, trains TranAD, and can optionally save a checkpoint and test reconstruction errors.
- `src/training/data.py` accepts either a legacy preprocessing artifact with a single `data` array or a newer artifact with precomputed `train_data`, `val_data`, and `test_data`; if splits are absent it falls back to config-driven chronological splitting.
- `src/training/configs/spt_tranad.yaml` is written for precomputed splits, but the code still supports legacy single-`data` artifacts via fallback splitting.
- `scripts/spt/run_preprocessing.py` is not a reliable entry point in the current tree because its default config depends on unregistered preprocessing steps.
- `scripts/spt/infer_spt_v2.py` is not part of that local path: it imports `anldq.*`, which is not present in this repository and not declared in `requirements.txt`.

## Verification

There is no verified repo-wide test/lint/typecheck command in this checkout.

Use focused checks instead of inventing a full validation pipeline:
- `bash setup.sh`
- `python scripts/atlas/fetch/fetch_one_run.py --help`
- `python scripts/spt/train_tranad.py --help`
- For preprocessing changes, use a repo-root import smoke test that loads a known config and constructs `PreprocessingPipeline`.

Do not claim lint, typecheck, or broad test coverage unless you actually find and run the corresponding command.

## Known Mismatches

Some prose docs still reference `scripts/atlas/fetch/fetch_one_run.py` as the quick-start path while older agent guidance referenced `scripts/atlas/fetch_one_run.py`. Trust the actual file tree and CLI under `scripts/atlas/fetch/`.

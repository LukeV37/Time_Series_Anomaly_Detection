# ATLAS Fetch

Fetch one or more ATLAS runs and write `merged.csv` output for each run.

## Setup

From the repo root:

```bash
source setup.sh
```

## Run

Fetch a single run:

```bash
python scripts/atlas/fetch/fetch_one_run.py --run-number 520705 --output-dir "$ATLAS_DATA_DIR"
```

Fetch multiple runs from a file:

```bash
python scripts/atlas/fetch/fetch_one_run.py --input-file scripts/atlas/fetch/runs.txt --output-dir "$ATLAS_DATA_DIR"
```

## Defaults

The script defaults to:

```text
src/atlas/pbeast_fetcher/configs/config.yaml
src/atlas/pbeast_fetcher/configs/sources.yaml
```

If `PBEAST_HTML_DIR` is set, it is used for run-summary HTML lookup.
Otherwise the script falls back to bundled package data under:

```text
src/atlas/pbeast_fetcher/data
```

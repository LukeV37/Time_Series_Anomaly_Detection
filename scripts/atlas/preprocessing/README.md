# ATLAS Preprocessing

Run the default ATLAS preprocessing pipeline on a fetched `merged.csv` file.

## Setup

From the repo root:

```bash
source setup.sh
```

## Run

Process a fetched run under `$ATLAS_DATA_DIR/<run_number>/merged.csv`:

```bash
python scripts/atlas/preprocessing/run_atlas_preprocessing.py 520705
```

Use an explicit CSV path instead:

```bash
python scripts/atlas/preprocessing/run_atlas_preprocessing.py 520705 --csv-path "$ATLAS_DATA_DIR/520705/merged.csv"
```

## Default Config

The script defaults to:

```text
src/preprocessing/configs/atlas_pipeline.yaml
```

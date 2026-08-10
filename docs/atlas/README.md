# ATLAS

This section documents the ATLAS-specific data access and preprocessing workflow used in this repository.

The broader goal is to support time-series anomaly detection on ATLAS operational data from monitoring compute nodes for data quality management. The intent is to detect abnormal behavior in the operational stack, including issues affecting compute nodes, data flow, and networking, so that problems can be identified and corrected more quickly. Faster detection helps reduce operational impact and improve the overall efficiency of the experiment.

The current documentation covers both the run-based PBeast fetch path and the initial CSV-based preprocessing path for merged ATLAS data.

## Quick Start

From the repo root:

```bash
source setup.sh
python scripts/atlas/fetch/fetch_one_run.py --run-number 520705 --output-dir "$ATLAS_DATA_DIR"
python scripts/atlas/preprocessing/run_atlas_preprocessing.py 520705
```

## Script Entry Points

- [`../../scripts/atlas/fetch/README.md`](../../scripts/atlas/fetch/README.md): how to fetch one or more runs into per-run `merged.csv` outputs
- [`../../scripts/atlas/preprocessing/README.md`](../../scripts/atlas/preprocessing/README.md): how to run the default ATLAS preprocessing pipeline on fetched `merged.csv` files

## Contents

- [`pbeast_fetcher.md`](./pbeast_fetcher.md): overview of the local PBeast fetch package, its configuration model, and how run-based data retrieval works in this repository
- [`time_series_alignment.md`](./time_series_alignment.md): overview of the reference-timeline alignment step used to convert fetched ATLAS time series into a single merged dataset for analysis

# PBeast Fetcher

## Purpose And Scope

`src/pbeast_fetcher/` contains the local PBeast access layer used by the ATLAS data-fetching workflow in this repository. It is a small, task-focused package that supports run-based retrieval of time-series data from PBeast and exposes the pieces needed by `scripts/ATLAS/fetch_one_run.py`.

This package is intentionally narrow in scope. It is designed around the current ATLAS workflow rather than as a general-purpose data access framework. In particular, it assumes the ATLAS TDAQ environment is available and that the Beauty client is provided by that environment rather than installed from PyPI.

## Package Structure

The package is organized around a few small modules with clear responsibilities.

- `src/pbeast_fetcher/__init__.py` exposes the minimal public surface currently used by downstream code: `PBeastFetcher` and `parse_run_summary`.
- `src/pbeast_fetcher/pbeast_fetcher.py` defines the main orchestration class. It handles configuration loading, connection lifecycle, source initialization, and fetching by run or by explicit time range.
- `src/pbeast_fetcher/beauty_client.py` wraps connection setup to the PBeast server through the Beauty library and applies environment-driven details such as proxy handling.
- `src/pbeast_fetcher/data_fetcher.py` manages retrieval for a single configured source and stores the returned series objects.
- `src/pbeast_fetcher/config_loader.py` reads the YAML configuration files and converts them into plain Python dictionaries used by the fetcher.
- `src/pbeast_fetcher/parsers.py` contains helpers for parsing run metadata, including the HTML run summaries used to resolve run numbers into start and end timestamps.
- `src/pbeast_fetcher/configs/config.yaml` contains server-level configuration such as URL, proxy, and timezone.
- `src/pbeast_fetcher/configs/sources.yaml` defines the named sources that can be fetched.
- `src/pbeast_fetcher/data/ATLASDataSummary*.html` provides the bundled run-summary inputs used by the run-based fetch path.

## Public API And Workflow

The main entry point is `PBeastFetcher`, typically constructed with `PBeastFetcher.from_config(...)` and used as a context manager.

At a high level, the run-based workflow is:

1. Load server configuration and source definitions from YAML.
2. Connect to PBeast through Beauty.
3. Resolve a run number into a concrete `(start, end)` time window using an ATLAS run summary HTML file.
4. Build one `DataFetcher` per enabled source.
5. Fetch all requested sources over the resolved time interval.
6. Return the fetched source objects to downstream code for alignment and export.

The `fetch_by_run(...)` path is the key integration point for the ATLAS scripts in this repository. It takes source names, a run year, a run number, and optionally an HTML summary path. If no HTML path is provided, the code falls back to the bundled `ATLASDataSummary{year}.html` file shipped with the package.

## Configuration Model

The configuration is intentionally simple.

`config.yaml` defines server-level settings used to initialize the Beauty client. The current loader recognizes fields for:

- server URL
- proxy
- timezone
- retry count

`sources.yaml` defines the available signals. Each source entry is flattened into a dictionary with the fields used by `DataFetcher`:

- `name`
- `category`
- `partition`
- `typ`
- `attr`
- `source`
- `regex`
- `enabled`
- `description`

This makes source selection configuration-driven rather than hardcoded in the package. The ATLAS script chooses which named sources to request, while the package remains responsible for translating the YAML definitions into Beauty queries.

One practical consequence is that a single configured source can expand into many returned series. This is especially important for regex-backed definitions such as the DCM source, where one logical source entry may produce a large set of channel-level time series.

## Returned Data Model

`fetch_by_run(...)` returns a mapping from source name to `DataFetcher` instance.

Each `DataFetcher` represents one configured source and stores the raw series objects returned by Beauty for the requested interval. Downstream code most commonly uses `get_all_data()` to retrieve the full list of series for that source.

In the current alignment workflow, these fetched objects are treated as named time series with datetime indices. That assumption is what allows `scripts/ATLAS/fetch_one_run.py` and `src/pbeast_fetcher/align.py` to build a single merged table across many channels.

## Run Time Resolution

Run-based fetching depends on translating a run number into an explicit time window. That logic lives in `src/pbeast_fetcher/parsers.py`.

`parse_run_summary(...)` reads an `ATLASDataSummary*.html` file, extracts run numbers and start/end timestamps, and normalizes the parsed times into the configured target timezone. `get_run_times(...)` then returns the `(start, end)` tuple for one run.

This repository keeps the HTML summaries under `src/pbeast_fetcher/data/` so the run lookup step is available locally and does not depend on a separate metadata service at runtime.

## Environment And Runtime Assumptions

The fetcher is not a standalone Python-only package. It depends on the ATLAS TDAQ runtime because the underlying `beauty` module is expected to come from the sourced release environment.

The operational flow in this repository is therefore:

1. configure environment overrides in `export.sh` if needed
2. build the local virtual environment with `scripts/ATLAS/setup.sh`
3. activate the runtime with `scripts/ATLAS/activate_atom.sh`
4. run `scripts/ATLAS/fetch_one_run.py`

The additional Python dependencies installed into the virtual environment are intentionally minimal and exist only to support this fetch-and-align workflow.

## Design Notes

This package is a minimal local copy tailored to the needs of this repository. The exported API is intentionally small, and the implementation is optimized for the run-based fetch path used by the ATLAS scripts rather than for broad feature coverage.

For the current workflow, the most important boundary is:

- `src/pbeast_fetcher/` is responsible for connecting to PBeast, resolving run windows, and returning the requested series
- `scripts/ATLAS/fetch_one_run.py` is responsible for choosing the sources to fetch, aligning them onto a common timeline, and writing the final CSV output

That split keeps the fetch layer focused on data access and leaves run-specific export behavior in the script layer.

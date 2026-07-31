# Time-Series Alignment

## Purpose

The ATLAS fetch workflow does not stop at retrieving raw PBeast series. For downstream analysis, the fetched signals need to be placed onto a single shared timeline so they can be written as one tabular dataset.

In the current workflow, `scripts/ATLAS/fetch_one_run.py` uses `L1ARate_Instant` as the reference timeline and aligns other fetched series onto it. This includes DCM channel series as well as additional scalar signals such as pileup and busy.

## Alignment Semantics

The alignment implemented in `src/pbeast_fetcher/align.py` uses backward as-of semantics.

For each reference timestamp `t` on the `L1ARate_Instant` timeline, the alignment step computes two quantities for every source series:

- the most recent source value observed at or before `t`
- `deltaT`, the age of that carried-forward value in seconds, defined as `t - sample_time`

This is equivalent to a forward-fill onto the reference grid, but expressed explicitly as an as-of join. The `deltaT` columns make the staleness of each aligned value visible instead of hiding it inside the merge logic.

## Inputs And Output Shape

The alignment step takes:

- one reference series
- zero or more additional source series

It produces a wide `pandas.DataFrame` with:

- a `timestamp` column for the reference timeline
- one value column per aligned signal
- one `*_deltaT` column per aligned signal

The reference signal itself is included in the output and receives a zero-valued `deltaT` column because it is already defined on the master timeline.

This output format is what `scripts/ATLAS/fetch_one_run.py` writes to `output/<run_number>/merged.csv`.

## Why The Alignment Path Changed

The original straightforward approach was to repeatedly apply `pandas.merge_asof` while growing a single wide DataFrame. That approach is simple and correct, but it becomes expensive when the number of source series is large.

This matters in the ATLAS use case because a single regex-backed source definition can expand into many concrete series. The DCM source is the main example: one logical fetch request may return a large number of channel-level time series. At that point, alignment cost can dominate the end-to-end runtime even after the fetch itself has completed.

## Available Strategies

Two alignment strategies are currently exposed through `src/pbeast_fetcher/align.py`.

### `baseline`

`baseline` is the direct reference implementation. It performs one `pandas.merge_asof` per source series onto a growing output frame.

This strategy is retained because it is easy to reason about and useful for correctness checks. It is not the preferred production path for wide outputs.

### `s2`

`s2` is the default fast path. It keeps the reference timeline fixed, uses `numpy.searchsorted` independently for each source series, computes values and `deltaT` arrays locally, and constructs the final wide DataFrame once at the end.

This avoids repeatedly expanding and copying an already-wide table. The implementation remains relatively compact, but scales much better when many channels are aligned onto the same reference timeline.

## Performance Rationale

At a high level, the difference between the two strategies is not the alignment rule itself but where the work is done.

`baseline` repeatedly merges into an ever-wider DataFrame, so each additional source series becomes more expensive than the last. `s2` treats each source series independently against the same fixed reference timestamps and delays DataFrame construction until all per-series arrays have been computed.

In practice, this keeps the fast path much closer to the actual problem structure: many separate series aligned against one shared clock.

The fast strategy was validated against the baseline on synthetic tests before being wired into the main script. For normal usage, `s2` should be treated as the default and `baseline` should be treated as a verification tool.

## Integration In The ATLAS Script

The integration point is `merged_dataframe_for_run(...)` in `scripts/ATLAS/fetch_one_run.py`.

That function:

1. reads the fetched `L1ARate_Instant` series
2. uses the first L1A series as the reference timeline
3. collects the remaining L1A series, DCM series, pileup series, and busy series as alignment inputs
4. dispatches to the selected alignment strategy
5. returns the final merged DataFrame for CSV export

The script exposes this choice through the `--merge-strategy` command-line option. The currently supported values are:

- `s2`
- `baseline`

The default is `s2`.

## Practical Usage Notes

For routine data production, use the default strategy.

```bash
python scripts/ATLAS/fetch_one_run.py --run-number 520479 --merge-strategy s2
```

Use `baseline` only when validating output equivalence or debugging an alignment issue.

```bash
python scripts/ATLAS/fetch_one_run.py --run-number 520479 --merge-strategy baseline
```

Large merged outputs are expected when broad regex source definitions are enabled. In particular, DCM expansion can produce many value and `deltaT` column pairs in a single file.

## Current Assumptions And Limits

The alignment logic currently assumes:

- one `L1ARate_Instant` series can serve as the master timeline for the run
- fetched source objects can be handled as named time series with datetime indices
- backward as-of semantics are the correct representation for downstream analysis

If the source definitions change, if the returned object shapes change, or if a different reference clock is required, the alignment contract should be revisited before extending the workflow.

## Code References

The relevant implementation points are:

- `src/pbeast_fetcher/align.py` for strategy definitions and the `STRATEGIES` registry
- `scripts/ATLAS/fetch_one_run.py` for reference-series selection, strategy dispatch, and CSV export
- `src/pbeast_fetcher/data_fetcher.py` for the fetched source containers that provide `get_all_data()` to the alignment step

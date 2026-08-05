"""Align many source series onto a single reference timeline.

Each source series is a pandas Series with a DatetimeIndex. For every reference
timestamp `t` we want, per source series:

  * value : the most recent source sample at or before `t`  (backward as-of / ffill)
  * deltaT: how stale that carried-forward value is, i.e. t - sample_time, in seconds

Two strategies are provided and share the same signature so they are
interchangeable:

    align(ref_series, source_series) -> pandas.DataFrame

  "s2"       fast: per-series numpy searchsorted, one DataFrame built at the end.
  "baseline" slow reference: one pd.merge_asof per series (O(M^2) in #series).
             Kept only for cross-checking correctness of "s2".

Select one via STRATEGIES[name].
"""

import numpy as np
import pandas as pd


def align_s2(ref_series, source_series):
    """Per-series numpy searchsorted against the fixed reference timeline.

    We never grow a wide frame inside the loop: each iteration produces two small
    1-D arrays (value + deltaT), and the DataFrame is assembled a single time.
    Cost is O(M * (k + N)) for M series, N reference points, k samples/series.
    """
    ref_name = ref_series.name or "ref"
    # Reference timestamps as int64 nanoseconds -> cheap integer math for deltaT.
    ref_ts = ref_series.index.values.astype("datetime64[ns]").view("int64")

    # Output starts with the reference value column and its (zero) deltaT.
    columns = {
        "timestamp": ref_series.index,
        ref_name: ref_series.to_numpy(),
        f"{ref_name}_deltaT": np.zeros(len(ref_ts)),
    }

    for index, item in enumerate(source_series):
        value_name = item.name or f"src_{index}"
        src_ts = item.index.values.astype("datetime64[ns]").view("int64")
        src_vals = item.to_numpy()

        # For each ref timestamp, index of the last source sample at or before it.
        # side="right" then -1 gives the rightmost sample with src_ts <= ref_ts.
        pos = np.searchsorted(src_ts, ref_ts, side="right") - 1
        valid = pos >= 0  # False where a ref timestamp precedes the first sample

        value_col = np.full(len(ref_ts), np.nan)
        value_col[valid] = src_vals[pos[valid]]

        delta_col = np.full(len(ref_ts), np.nan)
        # ns difference -> seconds; only where a prior sample exists.
        delta_col[valid] = (ref_ts[valid] - src_ts[pos[valid]]) / 1e9

        columns[value_name] = value_col
        columns[f"{value_name}_deltaT"] = delta_col

    # Single allocation of the final wide frame.
    return pd.DataFrame(columns)


def align_baseline(ref_series, source_series):
    """One pd.merge_asof per series onto a growing wide frame (slow reference).

    Semantically identical to align_s2 but grows the wide frame column-pair by
    column-pair, so it is O(M^2) in the number of series. Kept only to verify
    that the fast path produces the same numbers.
    """
    ref_name = ref_series.name or "ref"
    # Master timeline: the reference series as a sorted 'timestamp' column.
    df = ref_series.rename_axis("timestamp").reset_index().sort_values("timestamp")
    df[f"{ref_name}_deltaT"] = 0.0

    for index, item in enumerate(source_series):
        value_name = item.name or f"src_{index}"
        src_col = f"{value_name}_timestamp_src"
        # Turn this series into a 2-column frame [src_ts, value], sorted by time.
        item_df = item.rename_axis(src_col).reset_index().sort_values(src_col)
        # Backward as-of join: for each ref timestamp take the last sample <= it.
        df = pd.merge_asof(
            df, item_df, left_on="timestamp", right_on=src_col, direction="backward"
        )
        # Staleness of the carried-forward value on the reference timeline.
        delta = df["timestamp"] - df[src_col]
        df[f"{value_name}_deltaT"] = delta.dt.total_seconds()
        df = df.drop(columns=src_col)
    return df


# Registry: map a CLI-friendly name to its aligner function.
STRATEGIES = {
    "s2": align_s2,
    "baseline": align_baseline,
}

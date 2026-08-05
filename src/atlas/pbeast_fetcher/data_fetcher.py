"""
Data fetcher for a single PBeast source.
Fetches and stores data for one source configuration.
"""

import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def series_to_dataframe(ts, default_name, merge_on="timestamp"):
    """Convert one series (a Beauty timeseries or a DataFrame) to a DataFrame.

    Beauty series expose ``x`` (timestamps) and ``y`` (values). Returns an
    empty DataFrame if the series has no usable data.
    """
    if isinstance(ts, pd.DataFrame):
        df = ts.copy()
        if merge_on not in df.columns:
            df = df.reset_index().rename(columns={df.columns[0]: merge_on})
        return df

    timestamps = getattr(ts, "x", None)
    values = getattr(ts, "y", None)
    if timestamps is None or values is None:
        logger.warning("Timeseries '%s' missing x/y attributes; skipping", default_name)
        return pd.DataFrame()

    col_name = getattr(ts, "name", None) or default_name
    return pd.DataFrame({merge_on: list(timestamps), col_name: list(values)})


class DataFetcher:
    """
    Fetches and stores data for a single PBeast source.
    """
    
    def __init__(self, beauty_server, source_config):
        self._server = beauty_server
        self._config = source_config
        self._data = []
    
    @property
    def name(self):
        return self._config.get("name", "")
    
    @property
    def category(self):
        return self._config.get("category", "")
    
    @property
    def config(self):
        return self._config
    
    def fetch(self, since, till):
        try:
            fetch_start = time.time()
            timeseries = self._server.timeseries(
                since,
                till,
                self._config["partition"],
                self._config["typ"],
                self._config["attr"],
                source=self._config["source"],
                regex=self._config.get("regex", False),
                all_publications=True,
            )
            fetch_time = time.time() - fetch_start
            
            if not timeseries:
                logger.debug(f"No data for {self.name} (took {fetch_time:.1f}s)")
                return
            
            logger.info(f"Fetched {self.name}: {len(timeseries)} series in {fetch_time:.1f}s")
            for ts in timeseries:
                self._data.append(ts)
        
        except Exception as e:
            logger.warning(f"Failed to fetch {self.name}: {e}")
    
    def get_data(self, index=0):
        if not self._data or index >= len(self._data):
            return pd.DataFrame()
        return self._data[index]
    
    def get_all_data(self):
        return self._data
    
    def to_dataframes(self, prefix=None):
        frames = []
        for idx, ts in enumerate(self._data):
            df = series_to_dataframe(ts, f"{prefix or self.name}_{idx}")
            if not df.empty:
                frames.append(df)
        return frames
    
    def clear(self):
        self._data = []
    
    @property
    def is_empty(self):
        return not self._data

"""PBeast data fetcher.

``PBeastFetcher`` connects to PBeast (via the Beauty library) and fetches one
or more configured sources, by run number, by date range, or as a live/replay
stream. See the examples/ directory for end-to-end usage.
"""

import time
import logging
from datetime import datetime, timedelta
from importlib import resources

from .beauty_client import BeautyClient
from .data_fetcher import DataFetcher
from .config_loader import get_enabled_sources
from .parsers import get_run_times

logger = logging.getLogger(__name__)


def get_default_html_path(year):
    """Return the path to the bundled ATLASDataSummary{year}.html file."""
    resource = resources.files("pbeast_fetcher") / "data" / f"ATLASDataSummary{year}.html"
    if not resource.is_file():
        raise FileNotFoundError(
            f"Could not find ATLASDataSummary{year}.html in the pbeast_fetcher "
            "package data. Pass html_path explicitly."
        )
    return str(resource)

class PBeastFetcher:
    """
    Main interface for fetching PBeast data.
    """
    
    def __init__(self, server_config, sources_config):
        """
        Initialize PBeast fetcher.
        
        Args:
            server_config: Dictionary with server configuration
            sources_config: List of source configuration dictionaries
        """
        self.server_config = server_config
        self.sources_config = sources_config
        self.client = BeautyClient(server_config)
        self._fetchers = {}
        self._connected = False
    
    @classmethod
    def from_config(cls, config_path, sources_path):
        """
        Create fetcher from config and sources YAML files.
        
        Args:
            config_path: Path to config.yaml
            sources_path: Path to sources.yaml
            
        Returns:
            PBeastFetcher instance
        """
        from .config_loader import load_server_config, load_sources_config
        
        server_config = load_server_config(config_path)
        sources_config = load_sources_config(sources_path)
        return cls(server_config, sources_config)
    
    def connect(self):
        """Connect to PBeast and initialize sources."""
        self.client.connect()
        self._init_fetchers()
        self._connected = True
        return self
    
    def disconnect(self):
        """Disconnect from PBeast."""
        self.client.disconnect()
        self._fetchers.clear()
        self._connected = False
    
    def _init_fetchers(self):
        """Initialize data fetchers from config."""
        self._fetchers.clear()
        server = self.client.get_server()
        enabled_sources = get_enabled_sources(self.sources_config)
        
        for source_config in enabled_sources:
            name = source_config["name"]
            self._fetchers[name] = DataFetcher(server, source_config)
    
    def _get_fetchers_for_sources(self, source_names):
        """
        Get fetchers for specified source names.
        
        Args:
            source_names: List of source names or None for all
            
        Returns:
            Dictionary mapping source names to DataFetcher instances
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        if source_names is None:
            return self._fetchers
        
        fetchers = {}
        for name in source_names:
            if name in self._fetchers:
                fetchers[name] = self._fetchers[name]
            else:
                logger.warning(f"Source '{name}' not found, skipping")
        
        return fetchers
    
    def _fetch_all(self, fetchers, since, till):
        """
        Fetch data for all fetchers (no merging).
        
        Args:
            fetchers: Dictionary of source names to DataFetcher instances
            since: Start datetime
            till: End datetime
        """
        total_sources = len(fetchers)
        duration = (till - since).total_seconds() / 3600
        logger.info(f"Fetching {total_sources} source(s) for {duration:.1f} hour(s) ({since} to {till})")
        
        for i, (name, fetcher) in enumerate(fetchers.items(), 1):
            category = fetcher.category
            logger.info(f"[{i}/{total_sources}] Fetching {name} (category: {category})...")
            
            try:
                fetcher.clear()
                fetcher.fetch(since, till)
                logger.info(f"[{i}/{total_sources}] Completed {name}")
            except Exception as e:
                logger.error(f"[{i}/{total_sources}] Failed {name}: {e}")
                continue
        
        logger.info("All sources fetched.")
    
    def fetch_by_run(self, source_names, year, run_number, html_path=None):
        """
        Fetch data for given sources, year, and run number.
        Automatically looks up run times from HTML summary file.
        
        Args:
            source_names: List of source names to fetch
            year: Year of the run
            run_number: Run number
            html_path: Path to HTML summary file (default: data/ATLASDataSummary{year}.html)
            
        Returns:
            Dictionary mapping source names to DataFetcher instances
        """
        if html_path is None:
            html_path = get_default_html_path(year)
        
        run_times = get_run_times(
            html_path,
            run_number,
            target_timezone=self.server_config.get("timezone", "Europe/Zurich"),
            default_year=year,
        )
        
        if run_times is None:
            raise ValueError(f"Run {run_number} not found in HTML file: {html_path}")
        
        run_start, run_end = run_times
        logger.info(f"Run {run_number} times: {run_start} to {run_end}")
        
        fetchers = self._get_fetchers_for_sources(source_names)
        self._fetch_all(fetchers, run_start, run_end)
        return fetchers
    
    def fetch_by_date_range(self, source_names, since, till):
        """
        Fetch data for given sources with since and till dates/times.
        
        Args:
            source_names: List of source names to fetch
            since: Start datetime
            till: End datetime
            
        Returns:
            Dictionary mapping source names to DataFetcher instances
        """
        fetchers = self._get_fetchers_for_sources(source_names)
        self._fetch_all(fetchers, since, till)
        return fetchers
    
    def stream(
        self,
        source_names,
        interval=10,
        lookback=30,
        start_time=None,
        end_time=None,
        max_iterations=None,
        simulate_realtime=False,
    ):
        """
        Stream data in real-time (online mode) or replay historical data.
        """
        if interval <= 0:
            raise ValueError("interval must be > 0")
        if lookback <= 0:
            raise ValueError("lookback must be > 0")
        if start_time and end_time and end_time <= start_time:
            raise ValueError("end_time must be after start_time")
        
        if not self._connected:
            self.connect()
        
        fetchers = self._get_fetchers_for_sources(source_names)
        mode = "live" if start_time is None else ("replay-realtime" if simulate_realtime else "replay-fast")
        logger.info(
            f"Starting streaming for {len(fetchers)} source(s); "
            f"mode={mode}, interval={interval}s, lookback={lookback}s, "
            f"start_time={start_time}, end_time={end_time}, max_iterations={max_iterations}"
        )
        
        current_end = start_time or datetime.now()
        iterations = 0
        try:
            while True:
                poll_end = datetime.now() if start_time is None else current_end
                poll_start = poll_end - timedelta(seconds=lookback)
                
                self._fetch_all(fetchers, poll_start, poll_end)
                yield fetchers
                
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                
                if start_time is None:
                    time.sleep(interval)
                else:
                    if simulate_realtime:
                        time.sleep(interval)
                    current_end = current_end + timedelta(seconds=interval)
                    if end_time and current_end > end_time:
                        break
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        finally:
            logger.info("Streaming stopped")
    
    def get_source(self, name):
        """Get a specific data fetcher by name."""
        return self._fetchers.get(name)
    
    def get_sources_by_category(self, category):
        """Get all fetchers in a category."""
        result = {}
        for name, fetcher in self._fetchers.items():
            if fetcher.category == category:
                result[name] = fetcher
        return result
    
    @property
    def sources(self):
        """Get all source fetchers."""
        return self._fetchers
    
    @property
    def categories(self):
        """Get list of all categories."""
        cats = set()
        for source in self.sources_config:
            cats.add(source.get("category", ""))
        return list(cats)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

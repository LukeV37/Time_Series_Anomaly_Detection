"""
Beauty client wrapper for PBeast connection.
Manages connection to PBeast server via Beauty library.
"""

import os
import logging

logger = logging.getLogger(__name__)


class BeautyClient:
    """
    Simple wrapper for Beauty library connection to PBeast.
    """
    
    def __init__(self, config):
        """
        Initialize Beauty client.
        
        Args:
            config: Dictionary with server configuration (url, proxy, timezone, etc.)
        """
        self.config = config
        self._beauty = None
        self._proxy_was_set = False
    
    def connect(self):
        """Establish connection to PBeast server."""
        if self._beauty is not None:
            return
        
        if self.config.get("proxy") and "https_proxy" not in os.environ:
            os.environ["https_proxy"] = self.config["proxy"]
            self._proxy_was_set = True
        
        try:
            from beauty import Beauty
            self._beauty = Beauty(
                self.config["url"],
                timezone=self.config.get("timezone", "Europe/Zurich"),
                cookie_setup=Beauty.NOCOOKIE,
            )
            logger.info(f"Connected to PBeast: {self.config['url']}")
        except ImportError:
            raise ImportError("Beauty library not found. Source TDAQ release first.")
    
    def disconnect(self):
        """Close connection and cleanup."""
        if self._proxy_was_set and "https_proxy" in os.environ:
            del os.environ["https_proxy"]
            self._proxy_was_set = False
        self._beauty = None
    
    def get_server(self):
        """
        Get the underlying Beauty server instance.
        """
        if self._beauty is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._beauty
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

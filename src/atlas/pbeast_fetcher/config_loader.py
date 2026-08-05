"""
Configuration loader for PBeast fetcher.
Loads YAML configs and returns plain dictionaries.
"""

import yaml


def load_server_config(path):
    """
    Load server configuration from YAML file.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    
    server = data.get("server", {})
    if not server:
        server = data.get("pbeast_server", {})
    
    return {
        "url": server.get("server_url") or server.get("url", "https://pc-atlas-www.cern.ch"),
        "proxy": server.get("proxy", "atlasgw.cern.ch:3128"),
        "timezone": server.get("timezone", "Europe/Zurich"),
        "retry_count": server.get("retry_count", 3),
    }


def load_sources_config(path):
    """
    Load sources configuration from YAML file.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    
    sources = []
    for category, items in data.items():
        if isinstance(items, dict):
            for name, config in items.items():
                source = {
                    "name": name,
                    "category": category,
                    "partition": config["partition"],
                    "typ": config["typ"],
                    "attr": config["attr"],
                    "source": config["source"],
                    "regex": config.get("regex", False),
                    "enabled": config.get("enabled", True),
                    "description": config.get("description", ""),
                }
                sources.append(source)
    
    return sources


def get_enabled_sources(sources):
    return [s for s in sources if s.get("enabled", True)]


def get_sources_by_category(sources, category):
    return [s for s in sources if s.get("category") == category]


def get_source_by_name(sources, name):
    for source in sources:
        if source.get("name") == name:
            return source
    return None

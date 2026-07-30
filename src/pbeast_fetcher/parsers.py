"""
Parsers for GRL and HTML run summary files.
"""

import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from dateutil import parser as date_parser
import pytz

logger = logging.getLogger(__name__)


def parse_grl(grl_path):
    """
    Parse GRL XML file and extract runs with their accepted luminosity blocks.
    """
    tree = ET.parse(grl_path)
    root = tree.getroot()
    
    runs = {}
    for lumi_block in root.findall(".//LumiBlockCollection"):
        run_number = lumi_block.find("Run").text
        lumiblocks = []
        
        for lb_range in lumi_block.findall("LBRange"):
            start = int(lb_range.get("Start"))
            end = int(lb_range.get("End"))
            lumiblocks.extend(range(start, end + 1))
        
        runs[run_number] = lumiblocks
    
    return runs


def get_run_numbers_from_grl(grl_path):
    runs = parse_grl(grl_path)
    return [int(run) for run in runs.keys()]


def get_lbs_for_run(grl_path, run_number):
    runs = parse_grl(grl_path)
    return runs.get(str(run_number), [])


def parse_run_summary(html_path, target_timezone="Europe/Zurich", default_year=None):
    """
    Parse HTML file and extract run numbers with their start/end times.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    if default_year is None:
        filename = Path(html_path).name
        year_match = re.search(r"(\d{4})", filename)
        if year_match:
            default_year = int(year_match.group(1))
            logger.debug(f"Extracted year {default_year} from filename")
        else:
            default_year = datetime.now().year
            logger.warning(f"Could not extract year from filename, using {default_year}")
    
    runs = {}
    target_tz = pytz.timezone(target_timezone)
    run_pattern = r'<h3>Run\s+<a[^>]*>(\d+)</a></h3>.*?<tr><th>Start</th><td>([^<]+)</td></tr>.*?<tr><th>End</th><td>([^<]+)</td></tr>'
    
    matches = re.finditer(run_pattern, html_content, re.DOTALL)
    
    for match in matches:
        run_num = int(match.group(1))
        start_str = match.group(2).strip()
        end_str = match.group(3).strip()
        
        try:
            if str(default_year) not in start_str:
                start_str = f"{start_str} {default_year}"
            if str(default_year) not in end_str:
                end_str = f"{end_str} {default_year}"
            
            start_time_aware = date_parser.parse(start_str)
            end_time_aware = date_parser.parse(end_str)
            
            if start_time_aware.tzinfo is not None:
                start_time = start_time_aware.astimezone(target_tz).replace(tzinfo=None)
            else:
                start_time = start_time_aware
            
            if end_time_aware.tzinfo is not None:
                end_time = end_time_aware.astimezone(target_tz).replace(tzinfo=None)
            else:
                end_time = end_time_aware
            
            runs[run_num] = (start_time, end_time)
            logger.debug(f"Parsed run {run_num}: {start_time} to {end_time}")
            
        except Exception as e:
            logger.warning(f"Failed to parse times for run {run_num}: {e}")
            logger.debug(f"  Start: {start_str}, End: {end_str}")
            continue
    
    logger.info(f"Parsed {len(runs)} runs from HTML file")
    return runs


def get_run_times(html_path, run_number, target_timezone="Europe/Zurich", default_year=None):
    runs = parse_run_summary(html_path, target_timezone, default_year)
    return runs.get(run_number)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Platform ingestion script for Site from the Piano source.

This script:

- Loads lookup tables
- Loads API query definitions (Site_API)
- Iterates over all report-week combinations
- Runs Piano API requests
- Writes raw CSV snapshots to the raw data directory
- Runs ingestion QA tests
- Logs progress and issues
"""

from datetime import datetime, timedelta, timezone
import pandas as pd
pd.set_option('display.max_colwidth', None)

# --- Standard libs
import os
from pathlib import Path
import sys

# --- Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper.functions import load_and_validate_all_lookups
from helper import functions
from helper import test_functions
from helper.config import gam_info
from helper.security_config import api_key
from helper.logging_utils import setup_logger

logger = setup_logger(__name__)

PLATFORMID = "WWW"


# ======================================================================
# FUNCTION DEFINITIONS
# ======================================================================

def load_site_info(helper_dir):
    """
    Load the Site_API sheet and filter for entries relevant to this ingestion script, 
    so entries that result in reach figures relevant for GAM.

    Parameters
    ----------
    helper_dir : Path
        Path to the helper directory containing lookup files.

    Returns
    -------
    pandas.DataFrame
        Filtered site info table with Report No. and API definitions.
    """
    site_info = (
        pd.read_excel(helper_dir / gam_info["lookup_file"], sheet_name="Site_API")
          .drop(columns=["no results"])
    )
    site_info["Report No."] = site_info["Report No."].astype(str)
    site_info = site_info[site_info["script"] == "1_site_ingestion"]
    return site_info


def run_api_queries(site_info, time_codes, raw_dir):
    """
    Iterate over all report IDs and week ranges, if the relevant raw snapshots do not exist,
    execute Piano API calls, and write raw CSV snapshots.

    Parameters
    ----------
    site_info : pandas.DataFrame
        Table containing Report No., API URLs, and key references.
    time_codes : pandas.DataFrame
        GAM-time lookup table containing w/c and week numbers.
    raw_dir : Path
        Directory where raw CSV output is stored.
    """

    for _, site_row in site_info.iterrows():

        api_query = site_row["API"]
        api_query_key = api_key[site_row["api_key"]]
        report_no = site_row["Report No."]

        logger.info(f"Starting report {report_no}")

        for _, time_row in time_codes.iterrows():

            week_number = time_row["WeekNumber_finYear"]

            filename = raw_dir / (
                f"{gam_info['file_timeinfo']}_reportNo{report_no}_weekNo{week_number}.csv"
            )

            # Skip if file already exists
            if filename.exists():
                continue

            logger.info(f"... iterating {filename}")

            start = time_row["w/c"]
            end_ts = pd.to_datetime(start) + timedelta(days=6)
            end = end_ts.strftime("%Y-%m-%d")

            today = datetime.now(timezone.utc).date()
            if end_ts.date() > today:
                logger.info("Week ends in the future — stopping early.")
                break

            # Convert encoded Piano URL → query JSON
            query = functions.convert_url_to_query(api_query, start, end)

            # Execute API call
            temp = functions.api_call(query, api_query_key)

            # Annotate results
            temp["w/c"] = start
            temp["timestamp_queryRun"] = datetime.now().strftime("%y%m%d-%H%M")
            temp["API"] = api_query

            # Store results
            store_api_results(temp, filename, report_no)


def store_api_results(df, filename, report_no):
    """
    Store Piano API results to CSV or warn if empty.

    Parameters
    ----------
    df : pandas.DataFrame
        API return rows.
    filename : Path
        Output file path.
    report_no : str
        Report identifier for logging.
    """
    if df.empty:
        logger.warning(f"No data returned for report {report_no}")
    else:
        df.to_csv(filename, index=False)
        logger.info(f"Finished report {report_no}")


# ======================================================================
# MAIN (Sphinx-safe)
# ======================================================================

def main():
    """
    Main execution entry point for the automated site ingestion.

    **main():**

    1. load_site_info( )
    2. run_api_queries( ) 
    3. store_api_results( )
    """

    # ---------------------------
    # Paths
    # ---------------------------
    project_root = Path(__file__).resolve().parents[1]
    helper_dir = project_root / "helper"

    raw_dir = project_root / "data" / "raw" / PLATFORMID
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load lookups
    # ---------------------------
    lookups = load_and_validate_all_lookups(
        country_cols=["PlaceID", "ATI"],
        time_cols=["w/c", "WeekNumber_finYear"],
        test_script=f"{PLATFORMID}_1a",
    )

    time_codes = lookups["time_codes"]

    # ---------------------------
    # Load Site_API
    # ---------------------------
    site_info = load_site_info(helper_dir)

    # Initial test: ensure Report No. uniqueness
    test_functions.test_unique_entries(
        site_info,
        "Report No.",
        f"{PLATFORMID}_1a_13",
        "initial api query list",
    )

    # ---------------------------
    # Execute API calls
    # ---------------------------
    run_api_queries(site_info, time_codes, raw_dir)


# ======================================================================
# EXECUTION GUARD
# ======================================================================

if __name__ == "__main__":
    main()
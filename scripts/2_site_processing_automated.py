#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Platform processing script for Site.

Responsible for:

- Combining weekly raw ingestion files
- Schema validation and enrichment
- Joining to lookup tables
- Running join integrity tests
- Producing processed data outputs
"""

from datetime import datetime
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# ---------------------------
# Project imports
# ---------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
from helper.functions import load_and_validate_all_lookups
from helper import test_functions
from helper.config import gam_info
from helper.logging_utils import setup_logger

logger = setup_logger(__name__)
PLATFORMID = "WWW"


# =====================================================================
# FUNCTION DEFINITIONS (kept in execution order)
# =====================================================================

def load_raw_files(raw_dir, filename_key):
    """
    Load all raw Piano ingestion CSVs that match the expected filename pattern.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw ingestion CSVs.
    filename_key : str
        Prefix pattern used to identify valid raw files.

    Returns
    -------
    tuple
        combined_df : pandas.DataFrame
            Concatenated raw data.
        empty_files : list[str]
            Filenames that appear empty (only metadata columns).
        max_rows : int
            Maximum number of rows found in a single raw file (for QA reporting).
    """
    files = []
    empty_files = []
    max_rows = 0

    for file in tqdm(raw_dir.iterdir(), desc="Loading raw files"):
        if filename_key not in file.name:
            continue

        df = pd.read_csv(file)

        if df.shape[1] == 3:
            empty_files.append(file.name)

        max_rows = max(max_rows, df.shape[0])

        df["filename"] = file.name
        df["Report No."] = (
            file.name.split("_")[1]
            .replace("reportNo", "")
            .split("weekNo")[0]
        )

        files.append(df)

    return pd.concat(files), empty_files, max_rows

def run_week_coverage_test(combined_df, time_codes, site_info):
    """Wrapper around test_weeks_presence_per_account for readability."""
    _site_info = site_info.copy()
    _site_info['Start'] = gam_info['w/c_start']
    _site_info['End'] = gam_info['w/c_end']
    return test_functions.test_weeks_presence_per_account(
        key="w/c",
        channel_id_col=["Report No."],
        main_data=combined_df,
        week_lookup=time_codes[["w/c"]],
        channel_lookup=_site_info[["Report No.", "Start", "End"]],
        test_number=f"{PLATFORMID}_2a_14",
        test_step="Check all weeks present for each Report No."
    )

def add_query_context(full_df, site_info):
    """
    Add Space, PlatformID, ServiceID and further metadata from the lookup tables.

    Parameters
    ----------
    full_df : pandas.DataFrame
        Combined raw ingestion data.
    site_info : pandas.DataFrame
        API definition table, including PlatformID and ServiceID per Report No.

    Returns
    -------
    pandas.DataFrame
        Enriched dataset with report metadata applied.
    """
    test_functions.test_key_consistency(
        site_info,
        full_df,
        ["Report No.", "API"],
        f'{PLATFORMID}_2a_13',
        "adding report context info",
        focus="left",
    )
    return site_info.merge(full_df, on=["Report No.", "API"], how="inner")


def add_week_context(df, time_codes):
    """
    Add YearGAE and WeekNumber_finYear context from the lookup tables.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to enrich.
    time_codes : pandas.DataFrame
        GAM time lookup tables.

    Returns
    -------
    pandas.DataFrame
        Enriched dataset with financial-year week context.
    """
    week_lookup = time_codes.copy()
    week_lookup["w/c"] = pd.to_datetime(week_lookup["w/c"])

    return df.merge(
        week_lookup[["YearGAE", "WeekNumber_finYear", "w/c"]],
        on="w/c",
        how="left",
    )


def enrich_service(df, site_info, service_codes, service_language_map, non_js_map, app_map):
    """
    Apply all service enrichment layers:
    1. By Space (from site_info)
    2. By site_level2
    3. By language
    4. By producer_nonjs
    5. By app_name

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset after merging with Site_API.
    site_info : pandas.DataFrame
        Site_API lookup table.
    service_codes : pandas.DataFrame
        ServiceID lookup table.
    service_language_map : pandas.DataFrame
        Language → ServiceID mapping.
    non_js_map : pandas.DataFrame
        NonJS producer → ServiceID mapping.
    app_map : pandas.DataFrame
        Apps → ServiceID mapping.

    Returns
    -------
    pandas.DataFrame
        Fully enriched dataset with ServiceID resolved.
    """
    # 1. Service by Space
    test_functions.test_join_cardinality(
        df, site_info,
        ["Report No.", "Space"],
        f'{PLATFORMID}_2a_14',
        "space → service check",
    )
    df = df.merge(site_info[["Space", "ServiceID", "PlatformID"]], on="Space", how="inner")

    # 2. Service by site_level2
    df = df.merge(
        service_codes[["ServiceID", "site_level2"]].dropna(),
        on="site_level2",
        how="left",
        suffixes=("", "_y")
    )
    df["ServiceID"] = df["ServiceID"].fillna(df["ServiceID_y"])
    df.drop(columns=["ServiceID_y"], inplace=True)

    # 3. Service by language
    df = df.merge(
        service_language_map[["ServiceID", "language"]].dropna(),
        on="language",
        how="left",
        suffixes=("", "_y")
    )
    df["ServiceID"] = df["ServiceID"].fillna(df["ServiceID_y"])
    df.drop(columns=["ServiceID_y"], inplace=True)

    # 4. Service by producer_nonjs
    df = df.merge(
        non_js_map,
        on="producer_nonjs",
        how="left",
        suffixes=("", "_y")
    )
    df["ServiceID"] = df["ServiceID"].fillna(df["ServiceID_y"])
    df.drop(columns=["ServiceID_y"], inplace=True)

    # 5. Service by app_name
    if "app_name" in df.columns:
        df = df.merge(
            app_map[["app_name", "ServiceID"]],
            on="app_name",
            how="left",
            suffixes=("", "_y")
        )
        df["ServiceID"] = df["ServiceID"].fillna(df["ServiceID_y"])
        df.drop(columns=["ServiceID_y"], inplace=True)

    return df


def add_place_id(df, country_codes):
    """
    Add PlaceID based on geocountry (df) / ATI (country_codes) mapping, with cleaning rules applied.
    If PlaceID can't be identified the value is mapped to UNK (unknown).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset after service enrichment.
    country_codes : pandas.DataFrame
        PlaceID lookup table.

    Returns
    -------
    pandas.DataFrame
        Dataset with PlaceID added.
    """
    df["geo_country"] = df["geo_country"].fillna("Unknown").replace(
        "Europe (unknown country)", "Unknown"
    )
    country_codes = country_codes.rename(columns={'ATI': 'geo_country'})
    temp = df.merge(
        country_codes[["PlaceID", "geo_country"]],
        on="geo_country",
        how="left"
    )
    temp['PlaceID'] = temp['PlaceID'].fillna('UNK')
    return temp


def run_final_tests(df, country_codes, platform_codes, service_codes):
    """
    Run all final dataset integrity tests.

    Parameters
    ----------
    df : pandas.DataFrame
        Final enriched dataset.
    country_codes : pandas.DataFrame
    platform_codes : pandas.DataFrame
    service_codes : pandas.DataFrame
    """
    test_functions.test_key_consistency(
        df, country_codes, ["PlaceID"],
        f'{PLATFORMID}_2a_15', "final_placeID_check",
        focus='left'
    )

    test_functions.test_key_consistency(
        df, platform_codes, ["PlatformID"],
        f'{PLATFORMID}_2a_16', "final_platformID_check",
        focus='left'
    )

    test_functions.test_key_consistency(
        df, service_codes, ["ServiceID"],
        f'{PLATFORMID}_2a_17', "final_serviceID_check",
        focus='left'
    )


# =====================================================================
# MAIN (execution wrapper)
# =====================================================================

def main():
    """
    Main execution entry point for site processing.

    **main():**

    1. load lookups & raw ingestion files
    2. merge_with_site_info()
    3. add_week_context()
    4. enrich_service()
    5. add_place_id()
    6. cleaning
    7. run_final_tests()
    8. store in processed

    """

    # ---------------------------
    # Paths
    # ---------------------------
    project_root = Path(__file__).resolve().parents[1]
    helper_dir = project_root / "helper"
    raw_dir = project_root / "data" / "raw" / PLATFORMID
    processed_dir = project_root / "data" / "processed" / PLATFORMID
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load lookups
    # ---------------------------
    lookups = load_and_validate_all_lookups(
        service_cols=["ServiceID", "site_level2"],
        country_cols=["PlaceID", "ATI"],
        time_cols=["YearGAE", "w/c", "WeekNumber_finYear"],
        test_script=f'{PLATFORMID}_2a'
    )

    country_codes = lookups["country_codes"]
    platform_codes = lookups["platform_codes"]
    service_codes = lookups["service_codes"]
    time_codes = lookups["time_codes"]

    # ---------------------------
    # Load raw ingestion files
    # ---------------------------
    combined_df, empty_files, max_rows = load_raw_files(raw_dir, gam_info["file_timeinfo"])
    logger.info(f"Largest file contains: {max_rows} rows")
    logger.info(f"Empty raw files: {empty_files}")

    combined_df["w/c"] = pd.to_datetime(combined_df["w/c"])
        
    # ---------------------------
    # Merge with Site_API
    # ---------------------------
    site_info = pd.read_excel(helper_dir / gam_info["lookup_file"], sheet_name="Site_API")
    site_info = site_info[site_info['script'] == '1_site_ingestion']
    site_info["Report No."] = site_info["Report No."].astype(str)

    df = add_query_context(combined_df, site_info)
    logger.info(f"add query context {df.shape}")
    # ---------------------------
    # Raw weeks completeness test
    # ---------------------------

    run_week_coverage_test(combined_df, time_codes, site_info)
    logger.info(f"run week coverage test {df.shape}")
    # ---------------------------
    # Add financial week context
    # ---------------------------
    df = add_week_context(df, time_codes)
    logger.info(f"add week context{df.shape}")
    # ---------------------------
    # Keep essential columns
    # ---------------------------
    df = df[
        [
            "Category", "Report No.", "Space",
            "YearGAE", "WeekNumber_finYear", "w/c",
            "site_level2", "geo_country", "m_unique_visitors",
            "device_type", "language", "producer_nonjs"
        ]
    ]

    lookup_path = helper_dir / gam_info["lookup_file"]
    service_language_map = pd.read_excel(lookup_path, sheet_name="Site_language")
    non_js_map = pd.read_excel(lookup_path, sheet_name="Site_NonJS")
    app_map = pd.read_excel(lookup_path, sheet_name="Site_App")

    # ---------------------------
    # Service enrichment
    # ---------------------------
    df = enrich_service(df, site_info, 
                        service_codes, 
                        service_language_map, 
                        non_js_map, 
                        app_map)
    logger.info(f"enrich service {df.shape}")
    # ---------------------------
    # Country mapping
    # ---------------------------
    df = add_place_id(df, country_codes)
    logger.info(f"country mapping {df.shape}")
    # ---------------------------
    # Final cleaning
    # ---------------------------
    df["m_unique_visitors"] = pd.to_numeric(df["m_unique_visitors"], errors="coerce")

    df = df.drop(columns=["site_level2", "language", "producer_nonjs", "geo_country"])

    # ---------------------------
    # Final QA tests
    # ---------------------------
    run_final_tests(df, country_codes, platform_codes, service_codes)

    # ---------------------------
    # Save final output
    # ---------------------------
    df.to_csv(
        processed_dir / f"{gam_info['file_timeinfo']}_{PLATFORMID}_automated.csv",
        index=False
    )

    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
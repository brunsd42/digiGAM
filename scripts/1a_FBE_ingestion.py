#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Platform ingestion script for Facebook from the Piano source.

This script:

- Loads lookup tables
- Queries Redshift 
- Writes raw CSV snapshots to the raw data directory
- Runs ingestion QA tests
- Logs progress and issues
"""

import pandas as pd
pd.set_option('display.max_colwidth', None)

# --- Standard libs
from pathlib import Path
import sys

# --- Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper.functions import load_all_lookups
from helper import functions
from helper import test_functions
from helper.config import gam_info
from helper.logging_utils import setup_logger

logger = setup_logger(__name__)

PLATFORMID = "FBE"


# ======================================================================
# FUNCTION DEFINITIONS
# ======================================================================
def run_facebook_engagement_query(sql_query: str, raw_dir: Path, platform_id: str):
    """
    Execute the Facebook Redshift engagement query with a safe fallback.

    Steps:
    - Try to query Redshift using the given SQL.
    - If successful, save the result as the latest raw snapshot.
    - If Redshift fails, fall back to the last saved CSV.
    - Always return a dataframe with page_id prefixed by platform_id.
    """

    raw_file = raw_dir / f"{gam_info['file_timeinfo']}_{platform_id}_engagements_redshift_extract.csv"
    logger.info(f"Target raw file: {raw_file}")

    try:
        logger.info("Querying Redshift for Facebook engagements...")
        df = functions.execute_sql_query(sql_query)

        # Validate
        if df is None or df.empty:
            raise ValueError("Redshift query returned no data; falling back to cached extract.")

        # Standardise Channel ID
        df['page_id'] = platform_id + df['page_id'].astype(str)

        # Save fresh extract
        df.to_csv(raw_file, index=False, na_rep='')
        logger.info(f"Successfully saved Redshift extract → {raw_file}")

    except Exception as e:
        logger.warning(f"Redshift query failed ({e}). Using cached extract instead.")

        if not raw_file.exists():
            raise FileNotFoundError(
                f"Redshift query failed and no cached extract found at {raw_file}"
            )

        df = pd.read_csv(raw_file, keep_default_na=False)
        df['page_id'] = df['page_id'].astype(str)

        logger.info(f"Loaded cached extract → {raw_file}")

    logger.info(f"Loaded Facebook engagements: {df.shape}")
    return df

def standardise_facebook_ids(df: pd.DataFrame, platformID: str) -> pd.DataFrame:
    """
    Ensure Facebook Channel IDs follow GAM convention (prefix with platformID).
    Convert types and rename standard columns.
    """
    if not df["page_id"].astype(str).str.startswith(platformID, na=False).all():
        df["page_id"] = platformID + df["page_id"].astype(str)

    df = df.copy()
    df["page_id"] = df["page_id"].astype(str)
    df["week_commencing"] = pd.to_datetime(df["week_commencing"])

    df = df.rename(columns={
        "page_id": "Channel ID",
        "week_commencing": "w/c"
    })

    return df


def run_ingestion_QA(df: pd.DataFrame,
                     socialmedia_accounts: pd.DataFrame,
                     week_tester: pd.DataFrame,
                     platformID: str):
    """
    Run all Facebook ingestion QA tests:
    - All Channel IDs present
    - Complete week coverage for active accounts
    - No missing engaged_reach
    - No duplicate rows
    """
    channel_ids = socialmedia_accounts["Channel ID"].unique().tolist()

    # 1 — All Channel IDs present in SQL extract
    test_functions.test_filter_elements_returned(
        df, channel_ids, "Channel ID",
        f"{platformID}_1e_06",
        "Check that all page IDs are found in SQL"
    )

    # 2 — Week completeness per Channel ID
    test_functions.test_weeks_presence_per_account(
        key="w/c",
        channel_id_col=["Channel ID"],
        main_data=df,
        week_lookup=week_tester[["w/c"]],
        channel_lookup=socialmedia_accounts[["Channel ID", "Start", "End"]],
        test_number=f"{platformID}_1e_07",
        test_step="Check all weeks present for each account"
    )

    # 3 — Engaged reach present and positive / zero
    test_functions.test_non_null_and_positive(
        df,
        numeric_columns=["engaged_reach"],
        test_number=f"{platformID}_1e_08",
        test_step="Check engaged_reach contains no missing values"
    )

    # 4 — No duplicates
    test_functions.test_duplicates(
        df,
        ["Channel ID", "w/c"],
        test_number=f"{platformID}_1e_09",
        test_step="Check no duplicates in Redshift extract"
    )


def attach_services(df: pd.DataFrame,
                    socialmedia_accounts: pd.DataFrame,
                    platformID: str) -> pd.DataFrame:
    """
    Right-join to apply ServiceID.
    Applies test to ensure all lookup accounts exist in SQL.
    """
    merged = df.merge(
        socialmedia_accounts[["Channel ID", "ServiceID", "Start", "End"]],
        on="Channel ID",
        how="right",
        indicator=True
    )

    test_functions.test_key_consistency(
        df, socialmedia_accounts,
        ["Channel ID"],
        f"{platformID}_1e_10",
        "checking social media accounts in lookup, adding service",
        focus="right"
    )

    # Keep only weeks where account is active
    mask = (
        (merged["w/c"] >= merged["Start"])
        & (
            merged["End"].isna()
            | (merged["w/c"] <= merged["End"])
        )
    )

    return merged[mask]


def store_processed(df: pd.DataFrame, platformID: str) -> None:
    """
    Save processed Facebook ingestion output to /data/processed/.
    """
    output_dir = Path(f"../data/processed/{platformID}")
    output_dir.mkdir(exist_ok=True, parents=True)

    cols = ["Channel ID", "ServiceID", "w/c", "engaged_reach"]

    out_path = output_dir / f"{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv"
    df[cols].to_csv(out_path, index=False)

    logger.info(f"Saved processed ingestion → {out_path}")

# ======================================================================
# MAIN
# ======================================================================

def main():
    """
    Main execution entry point for the facebook engagement ingestion.

    **main():**

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
    lookups = load_all_lookups(
        gam_info=gam_info,
        platformID=PLATFORMID,
        script='1a',
        with_country=False,
        with_service=False,
    )
    
    time_codes = lookups["time_codes"]
    socialmedia_accounts = lookups['socialmedia_accounts']

    # ---------------------------
    # SQL Query
    # ---------------------------
    FACEBOOK_ENGAGEMENT_SQL = f"""
        SELECT
            week_commencing,
            page_id,
            CASE
                WHEN (AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user)
                    > AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer)
                THEN ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user))
                    + (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer))*0.04827
                ELSE (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer))
                    + ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user))*0.04822
            END AS engaged_reach
        FROM 
            redshiftdb.central_insights.adverity_social_facebook_by_page AS p
        RIGHT JOIN
            world_service_audiences_insights.social_media_lookup_fb AS l
            ON p.page_id = l.fb_page_id
        WHERE 
            period = 'week'
            AND week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
        GROUP BY 
            week_commencing, page_id
    ;
    """

    facebook_engagements_raw = run_facebook_engagement_query(
        FACEBOOK_ENGAGEMENT_SQL,
        raw_dir=raw_dir,
        platform_id=PLATFORMID
    )
    # ---------------------------
    # Cleaning
    # ---------------------------
    facebook_engagements_raw = standardise_facebook_ids(
        facebook_engagements_raw,
        PLATFORMID
    )

    run_ingestion_QA(
        facebook_engagements_raw,
        socialmedia_accounts=socialmedia_accounts,
        week_tester=time_codes,
        platformID=PLATFORMID
    )

    facebook_engagements = attach_services(
        facebook_engagements_raw,
        socialmedia_accounts=socialmedia_accounts,
        platformID=PLATFORMID
    )

    store_processed(facebook_engagements, PLATFORMID)

# ======================================================================
# EXECUTION GUARD
# ======================================================================
if __name__ == "__main__":
    main()
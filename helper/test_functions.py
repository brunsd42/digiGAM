"""
Shared test utilities for ingestion & processing pipelines.

These functions validate:
- lookup table integrity
- key presence and join consistency
- duplicate keys
- missing values
- join cardinality (row explosion)
- uniqueness constraints

All tests log results and update a daily Excel-based logbook.

"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook

from typing import List #, Optional, Dict, Set

from helper.logging_utils import setup_logger
logger = setup_logger(__name__)

# points to: GAM_pivot/helper/GAM_Lookup.xlsx
LOOKUP_XLSX = Path(__file__).resolve().parent / "GAM_Lookup.xlsx"

# =============================================================================
# LOOKUP TESTS
# =============================================================================

def test_lookup_files(df, id_columns, test_numbers, test_step):
    """
    Run the three standard lookup integrity tests:
    
    1. `test_not_empty` — ensure the lookup table is not empty  
    2. `test_no_duplicates` — ensure the ID columns contain unique keys  
    3. `test_no_missing_values` — ensure required ID columns contain no nulls

    Parameters
    ----------
    df : pandas.DataFrame
        The lookup table being tested.
    id_columns : list[str]
        Columns that must be unique and non-null.
    test_numbers : list[str]
        Three test-number identifiers, corresponding to the three tests.
    test_step : str
        Description of where the lookup is being validated (for logbook tracking).
    """
    test_not_empty(df, test_numbers[0], test_step=test_step, name="lookup")
    test_no_duplicates(df, id_columns, test_numbers[1], test_step=test_step)
    test_no_missing_values(df, id_columns, test_numbers[2], test_step=test_step, name="lookup")


def test_not_empty(df, test_number, test_step="", name="lookup"):
    """
    Verify that a DataFrame is not empty.

    Logs the result and records the outcome in the daily test logbook.
    This is used across ingestion and processing pipelines to ensure
    lookup tables and intermediate datasets contain at least one row.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to test.
    test_number : str
        Unique test identifier for tracking in the logbook.
    test_step : str, optional
        Description of the pipeline step running this test.
    name : str, optional
        Friendly name used in log messages (default `"lookup"`).
    """
    issues = pd.DataFrame()

    if df.empty:
        logger.error(f"❌ Test {test_number} failed: {name} DataFrame is empty.")
        issues = pd.DataFrame({"Issue": [f"{name} DataFrame is empty"]})
    else:
        logger.info(f"✅ Test {test_number} passed: {name} DataFrame is not empty.")

    update_logbook(test_number, issues, test=f"test_not_empty ({name})", test_step=test_step)


def test_no_duplicates(df, columns, test_number, test_step=""):
    """
    Verify that no duplicate key combinations exist in the specified columns.

    This test ensures that lookup tables and enriched datasets maintain
    the expected uniqueness constraints on their key columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.
    columns : list[str]
        Columns that together must form a unique key.
    test_number : str
        Unique identifier for logging and issue-list tracking.
    test_step : str, optional
        Description of the pipeline step where this test is executed.
    """
    issues = (
        df.groupby(columns, dropna=False)
          .size()
          .reset_index(name="Count")
          .query("Count > 1")
    )

    if issues.empty:
        logger.info(f"✅ Test {test_number} passed: No duplicate {columns}.")
    else:
        logger.error(f"❌ Test {test_number} failed: Duplicate {columns} found.")

    update_logbook(test_number, issues, test="test_no_duplicates", test_step=test_step)


def test_no_missing_values(df, columns, test_number, test_step="", name="lookup"):
    """
    Verify that none of the specified key columns contain missing values.

    Used to validate lookup tables, join keys, and enriched fields before
    downstream joins or aggregations.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to evaluate.
    columns : list[str]
        Columns that must contain no null or missing values.
    test_number : str
        Test identifier for logbook tracking.
    test_step : str, optional
        Logical pipeline step for test reporting.
    name : str, optional
        Friendly name used in log messages (default `"lookup"`).
    """
    missing_counts = df[columns].isnull().sum()
    issues = pd.DataFrame()

    if missing_counts.any():
        missing_detail = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}
        logger.error(f"❌ Test {test_number} failed: Missing values detected in {name}: {missing_detail}")
        issues = pd.DataFrame({"Issue": [f"Missing values: {missing_detail}"]})
    else:
        logger.info(f"✅ Test {test_number} passed: No missing values in {name}.")

    update_logbook(test_number, issues, test=f"test_no_missing_values ({name})", test_step=test_step)

# =============================================================================
# SITE-SPECIFIC TESTS
# =============================================================================

def test_unique_entries(df, column_name, test_number, test_step=""):
    """
    Verify that all values in a specified column are unique.

    This is primarily used for validating ingestion configuration tables,
    such as ensuring that each report number in `Site_API` appears only
    once.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to test.
    column_name : str
        Column whose uniqueness must be enforced.
    test_number : str
        Identifier used in logs and issue tracking.
    test_step : str, optional
        Description of the pipeline stage performing the test.
    """
    duplicates = df[df[column_name].duplicated()][column_name]

    if duplicates.empty:
        logger.info(f"✅ Test {test_number} passed: All values in '{column_name}' are unique.")
    else:
        logger.error(f"❌ Test {test_number} failed: Duplicate values detected in '{column_name}'.")
        logger.info(duplicates.to_string(index=False))

    update_logbook(test_number, duplicates, test="test_unique_entries", test_step=test_step)

# =============================================================================
# JOIN INTEGRITY TESTS
# =============================================================================

def test_key_consistency(df_left, df_right, key, test_number, test_step="", focus="both", logger=None):
    """
    Verify that df_left and df_right contain matching key values.

    Parameters
    ----------
    df_left : pandas.DataFrame
        Left dataset for comparison.
    df_right : pandas.DataFrame
        Right dataset for comparison.
    key : str or list[str]
        Join key(s).
    test_number : str
        Unique identifier for logging.
    test_step : str, optional
        Description of where in the pipeline the test is executed.

    Notes
    -----
    Writes issues to the daily logbook automatically.
    """
    log = logger.info if hasattr(logger, "info") else print

    key_cols = key if isinstance(key, list) else [key]

    merged = df_left[key_cols].merge(
        df_right[key_cols],
        on=key_cols,
        how="outer",
        indicator=True
    )

    issue_left = merged[merged["_merge"] == "left_only"] if focus != "right" else pd.DataFrame()
    issue_right = merged[merged["_merge"] == "right_only"] if focus != "left" else pd.DataFrame()

    if issue_left.empty and issue_right.empty:
        log(f"✅ Test {test_number} passed: No key mismatches.")
    else:
        log(f"❌ Test {test_number} failed: Key mismatches found.")
        if not issue_left.empty:
            log(f"   Left-only: {issue_left.shape[0]}")
        if not issue_right.empty:
            log(f"   Right-only: {issue_right.shape[0]}")

    issues = pd.concat([issue_left, issue_right], ignore_index=True).drop_duplicates()

    update_logbook(
        test_number,
        issues,
        test="test_key_consistency",
        test_step=test_step
    )


def test_join_cardinality(df_left, df_right, key, test_number, test_step="", logger=None):
    """
    Ensure that joining df_left to df_right does not increase row count.

    Detects 1→many explosions that would otherwise corrupt metrics.

    Parameters
    ----------
    df_left : pandas.DataFrame
        Baseline dataset.
    df_right : pandas.DataFrame
        Enrichment or lookup dataset.
    key : str or list[str]
        Join key(s).
    test_number : str
        Unique identifier for logging.
    test_step : str, optional
        Description of pipeline stage.
    """
    log = logger.info if hasattr(logger, "info") else print

    key_cols = key if isinstance(key, list) else [key]

    before = df_left.shape[0]

    merged = df_left[key_cols].merge(
        df_right[key_cols],
        on=key_cols,
        how="left",
        indicator=True
    )

    after = merged.shape[0]
    issues = pd.DataFrame()

    if before == after:
        log(f"✅ Test {test_number} passed: Join cardinality preserved.")
    else:
        dup_mask = merged.duplicated(subset=key_cols, keep=False)
        issues = merged[dup_mask]
        log(f"❌ Test {test_number} failed: {issues.shape[0]} rows show join explosion.")

    update_logbook(
        test_number,
        issues,
        test="test_join_cardinality",
        test_step=test_step
    )

def test_weeks_presence_per_account(
    key: str,
    channel_id_col: list,
    main_data: pd.DataFrame,
    week_lookup: pd.DataFrame,
    channel_lookup: pd.DataFrame,
    test_number,
    test_step: str = ''
):
    """
    Check that each channel has all expected weeks present from its Start date onward.

    Parameters
    ----------
    key : str
        Week column name (e.g., 'w/c', 'Week Number').
    channel_id_col : list[str]
        Column(s) that uniquely identify the channel/account/report.
    main_data : pd.DataFrame
        Observed dataset containing actual weeks.
    week_lookup : pd.DataFrame
        Table listing all canonical weeks.
    channel_lookup : pd.DataFrame
        Table containing channel start (and optional end) dates.
    test_number : str
        Identifier passed to logbook.
    test_step : str
        Free text describing pipeline step.

    Returns
    -------
    pd.DataFrame
        Rows representing missing (channel_id, week) combinations.
    """
    main_test_data = main_data.copy()
    week_lookup_test_data = week_lookup.copy()
    start_dates = channel_lookup.copy()
    today = pd.Timestamp.today().normalize()

    # ---------------------------
    # 1. Validate schema
    # ---------------------------
    for df_name, df, cols in [
        ('main_data', main_test_data, channel_id_col + [key]),
        ('week_lookup', week_lookup_test_data, [key]),
        ('channel_lookup', start_dates, channel_id_col + ['Start', 'End']),
    ]:
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} missing columns: {missing_cols}")

    # ---------------------------
    # 2. Convert to datetime
    # ---------------------------
    if key != 'Week Number':
        main_test_data[key] = pd.to_datetime(main_test_data[key], errors='coerce', dayfirst=True)
        week_lookup_test_data[key] = pd.to_datetime(week_lookup_test_data[key], errors='coerce', dayfirst=True)
    else:
        main_test_data[key] = pd.to_numeric(main_test_data[key], errors='coerce')
        week_lookup_test_data[key] = pd.to_numeric(week_lookup_test_data[key], errors='coerce')

    main_test_data = main_test_data.dropna(subset=[key])
    week_lookup_test_data = week_lookup_test_data.dropna(subset=[key])
    start_dates = start_dates.dropna(subset=['Start'])

    # ---------------------------
    # 3. Restrict week_lookup to fully completed weeks
    # ---------------------------
    if key != 'Week Number':
        last_monday_prev_week = today - pd.Timedelta(days=today.weekday() + 7)
        week_lookup_test_data = week_lookup_test_data[week_lookup_test_data[key] <= last_monday_prev_week]

    # ---------------------------
    # 4. Build expected grid: (channel × week)
    # ---------------------------
    start_dates['_tmp'] = 1
    week_lookup_test_data['_tmp'] = 1

    expected = (
        start_dates.merge(week_lookup_test_data, on='_tmp', how='inner')
        .drop(columns=['_tmp'])
    )

    expected = expected[expected[key] >= expected['Start']]
    expected = expected[channel_id_col + ['Start', 'End', key]].drop_duplicates()

    # ---------------------------
    # 5. Actual data: de-duplicate
    # ---------------------------
    actual = main_test_data[channel_id_col + [key]].drop_duplicates()

    # ---------------------------
    # 6. Missing = expected - actual
    # ---------------------------
    missing = expected.merge(
        actual,
        on=channel_id_col + [key],
        how='left',
        indicator=True
    )
    missing = missing[missing['_merge'] == 'left_only']
    missing = missing.drop(columns=['_merge'])
    missing = missing[
        (missing[key] >= missing['Start']) &
        (missing[key] <= missing['End'].fillna(today))
    ].sort_values(channel_id_col + [key])

    # ---------------------------
    # 7. Summary to log
    # ---------------------------
    summary = (
        missing.groupby(channel_id_col)[key]
        .nunique()
        .reset_index(name='missing_week_count')
        .sort_values('missing_week_count', ascending=False)
    )

    logger.info("\nSummary of missing weeks per channel:")
    logger.info("\n" + summary.to_string(index=False))

    # ---------------------------
    # 8. Log to issue logbook
    # ---------------------------
    update_logbook(
        test_number,
        missing,
        test='missing weeks by channel since Start',
        test_step=test_step
    )

    return missing

def test_hierarchy_reach(
    test_number,
    mode,
    gam_info,
    df,
    key,
    metric_col,
    test_step,
    round_metric=False
):
    """
    Test that each parent service/platform has Reach >= Reach of each of its
    child services/platforms. A parent may be smaller than the *sum* of its
    children but must not be smaller than any individual child.

    Parameters
    ----------
    test_number : str
        Identifier for logbook entry.
    mode : {'Service', 'Platform'}
        Which hierarchy to validate.
    gam_info : dict
        Configuration dictionary, must contain the lookup file.
    df : pd.DataFrame
        Must contain columns matching mode ('ServiceID' or 'PlatformID')
        plus Reach and the grouping keys.
    key : list[str]
        Columns that define the reach subgroup (e.g. ['w/c','PlaceID']).
    metric_col : str
        Name of the metric to compare (e.g., 'Reach').
    test_step : str
        Description for logging.
    round_metric : bool, optional
        Whether to round parent/child metrics before comparison.

    Returns
    -------
    pd.DataFrame
        All failing cases, empty DataFrame if none.
    """

    # --------------------------------------
    # Sanity checks
    # --------------------------------------
    if mode not in ["Service", "Platform"]:
        raise ValueError("mode must be 'Service' or 'Platform'.")

    if not isinstance(key, list):
        raise ValueError("key must be a list of column names.")

    if any(col not in df.columns for col in key):
        raise KeyError(f"Some key columns missing: {key}")

    # --------------------------------------
    # Determine hierarchy sheet and ID column
    # --------------------------------------
    if mode == "Service":
        sheet_name = "Service Hierarchy"
        id_col = "ServiceID"
        dimension = "PlatformID"
    else:
        sheet_name = "Platform Hierarchy"
        id_col = "PlatformID"
        dimension = "ServiceID"

    # --------------------------------------
    # Load hierarchy
    # --------------------------------------
    
    hierarchy_df = pd.read_excel(
        LOOKUP_XLSX,
        sheet_name=sheet_name,
        engine="openpyxl"
    )[['Parent', 'Child']].dropna()

    # Keep only hierarchy relevant to df
    hierarchy_df = hierarchy_df[
        hierarchy_df['Parent'].isin(df[id_col]) &
        hierarchy_df['Child'].isin(df[id_col])
    ]

    if hierarchy_df.empty:
        logger.info(f"No applicable hierarchy found for mode={mode}.")
        return pd.DataFrame()

    # --------------------------------------
    # Build descendant tree (multi-level)
    # --------------------------------------
    def get_descendants(mapping, parent):
        descendants = set()
        stack = [parent]
        while stack:
            cur = stack.pop()
            children = mapping[mapping['Parent'] == cur]['Child'].tolist()
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
        return descendants

    expanded = []
    for parent in hierarchy_df['Parent'].unique():
        desc = get_descendants(hierarchy_df, parent)
        for child in desc:
            expanded.append({'Parent': parent, 'Child': child})

    if not expanded:
        logger.info("Hierarchy exists but no descendants found.")
        return pd.DataFrame()

    expanded_df = pd.DataFrame(expanded)

    # --------------------------------------
    # Compare parent vs child
    # --------------------------------------
    issues = []

    for subgroup in df[dimension].unique():
        temp = df[df[dimension] == subgroup]

        merged = (
            expanded_df
            .merge(temp, left_on="Child", right_on=id_col, how="inner")
            .rename(columns={metric_col: "Child_val"})
            .drop(columns=id_col)
            .merge(
                temp,
                left_on=key + ["Parent"],
                right_on=key + [id_col],
                how="inner",
            )
            .rename(columns={metric_col: "Parent_val"})
            .drop(columns=id_col)
        )

        if round_metric:
            merged["Child_val"] = merged["Child_val"].round()
            merged["Parent_val"] = merged["Parent_val"].round()

        fail = merged[merged["Child_val"] > merged["Parent_val"]].copy()
        if not fail.empty:
            fail["diff"] = fail["Child_val"] - fail["Parent_val"]
            issues.append(fail)

    # --------------------------------------
    # Compile results
    # --------------------------------------
    if issues:
        full = pd.concat(issues).sort_values("diff", ascending=False)
        update_logbook(test_number, full, f"hierarchy reach test", test_step)
        print("❌ Hierarchy test failed - see issues above.")
        return full

    update_logbook(test_number, pd.DataFrame(), f"hierarchy reach test", test_step)
    print("✅ All hierarchy tests passed.")
    return pd.DataFrame()

def test_duplicates(df: pd.DataFrame, columns: List[str], test_number: str, test_step: str = "") -> None:
    """
    Fail if any combination of the key columns appears more than once.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame under test.
    columns : list[str]
        Columns that define a unique key.
    test_number : str
        Unique test identifier.
    test_step : str, optional
        Context string for logging.
    """
    # Group with dropna=False to treat NaN as a distinct key value (safer for lookups)
    issues_df = (
        df.groupby(columns, dropna=False)
          .size()
          .reset_index(name="Count")
    )
    issues_df = issues_df[issues_df["Count"] > 1]

    if not issues_df.empty:
        print(f"❌ Test {test_number} failed: The following combinations occur more than once")
        # Optional: print a small preview for quick debugging
        print(issues_df.head(10).to_string(index=False))
    else:
        print(f"✅ Test {test_number} passed: No combinations occurs more than once.")

    update_logbook(
        test_number=test_number,
        issues_list=issues_df,
        test="Testing the combination of columns for uniqueness",
        test_step=test_step,
    )


# =============================================================================
# LOGBOOK
# =============================================================================

def update_logbook(test_number, issues_list, test='', test_step=''):
    """
    Update the daily test logbook and optional issue list.
    Minimal fix version (KISS/DRY).
    """

    logger.info('...updating logbook...\n')
    today_date = datetime.now().strftime('%Y-%m-%d')

    # ---------------------------
    # Paths
    # ---------------------------
    project_root = Path(__file__).resolve().parents[1]
    test_dir = project_root / "test" / f"issue_lists_{today_date}"
    test_dir.mkdir(parents=True, exist_ok=True)

    logbook_path = test_dir / "_test_logbook.xlsx"

    # ---------------------------
    # Create logbook if missing
    # ---------------------------
    if not logbook_path.exists():
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        headers = ['test number', 'timestamp', 'pass/fail',
                   'check file', 'test', 'step']
        ws.append(headers)
        wb.save(logbook_path)

    # ---------------------------
    # Load sheet
    # ---------------------------
    logbook_df = pd.read_excel(logbook_path, sheet_name='Sheet1', engine='openpyxl')

    # Ensure required columns exist
    for col in ['test number', 'timestamp', 'pass/fail',
                'check file', 'test', 'step']:
        if col not in logbook_df.columns:
            logbook_df[col] = ""

    # Remove previous entry for this test
    logbook_df = logbook_df[logbook_df['test number'] != test_number]

    # ---------------------------
    # Append fresh row
    # ---------------------------
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row = {
        'test number': test_number,
        'timestamp': timestamp,
        'pass/fail': '',
        'check file': '',
        'test': test,
        'step': test_step
    }
    logbook_df = pd.concat([logbook_df, pd.DataFrame([new_row])], ignore_index=True)

    # ---------------------------
    # Issues file
    # ---------------------------
    if issues_list is not None and hasattr(issues_list, "empty") and not issues_list.empty:
        file_name = test_dir / f"{test_number}_issue_list.csv"
        pd.DataFrame(issues_list).to_csv(file_name, index=False)
        logbook_df.loc[logbook_df['test number'] == test_number, 'pass/fail'] = 'fail'
        logbook_df.loc[logbook_df['test number'] == test_number, 'check file'] = file_name.name
    else:
        logbook_df.loc[logbook_df['test number'] == test_number, 'pass/fail'] = 'pass'
        logbook_df.loc[logbook_df['test number'] == test_number, 'check file'] = 'no file created :)'

    # ---------------------------
    # Save back (replace sheet)
    # ---------------------------
    with pd.ExcelWriter(logbook_path, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        logbook_df.to_excel(writer, sheet_name='Sheet1', index=False)




















# Test to check if all filter elements were returned
def test_filter_elements_returned(df, filter_elements, column_name, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    print(f"...testing {column_name}...")
    returned_elements = df[column_name].unique().tolist()
    missing_elements = set(filter_elements) - set(returned_elements)
    issues_df = pd.DataFrame(list(missing_elements), columns=[column_name])
    
    # Print the results
    if not issues_df.empty:
        print(f"❌ Test {test_number} failed: not all elements from lookup were retrieved in dataset - decide if they are really missing or if these are inactive ")
    else:
        print(f"✅ Test {test_number} passed: everything found!")
    
    update_logbook(test_number=test_number, issues_list= issues_df, 
                   test_step=test_step, test='testing missing elements in columns')

# Test to check if the country percentage adds up to a 100% #former test_country_percentage
def test_percentage(df, groupby_columns, test_number, test_step, percentage_col='country_%'):
    '''
    # sphinx-autodoc-skip
    '''
    test_df = df.copy()
    
    # Group by fb_page_name and fb_metric_end_time, and sum country_%
    test_df[percentage_col] = test_df[percentage_col] * 100
    test_df = test_df.groupby(groupby_columns)[percentage_col].sum().reset_index()
    test_df[percentage_col] = test_df[percentage_col].round(0)
    issues_df = test_df[test_df[percentage_col] != 100.0]

    update_logbook(test_number, issues_df, 'testing country percentage', test_step)

def test_non_null_and_positive(df, numeric_columns=None, test_number='', test_step=''):
    """
    # sphinx-autodoc-skip
    
    Test that the numeric columns of the dataframe have no NaN values and contain values > 0.

    Parameters:
        df (pd.DataFrame): DataFrame to check.
        numeric_columns (list): List of numeric columns to check for > 0. If None, checks all numeric columns.
        test_number (str): Unique test identifier.
        test_step (str): Optional description of the test step.

    """
    issues_df = pd.DataFrame()

    # Check for NaN values in numeric columns
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        print('numeric columns was not indicated - applying test to all numeric columns')

    nan_issues = df[df[numeric_columns].isnull().any(axis=1)]
    if not nan_issues.empty:
        print(f"❌ Test {test_number} failed: NaN values detected in numeric columns.")
        issues_df = pd.concat([issues_df, nan_issues])

    # Check for non-positive values
    for col in numeric_columns:
        non_positive = df[df[col] < 0]
        if not non_positive.empty:
            print(f"❌ Test {test_number} failed: Non-positive values detected in column '{col}'.")
            issues_df = pd.concat([issues_df, non_positive])

    if issues_df.empty:
        print(f"✅ Test {test_number} passed: No NaN and all numeric values > 0.")
    else:
        issues_df = issues_df.drop_duplicates()

    # Log results
    update_logbook(test_number, issues_df, test='Check for NaN and positive values', test_step=test_step)



#############################################################################################################
# test same values
def test_allowed_values(df, test_column, allowed_values, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    # Check if any country/channel occurs more than once a week
    issues_df = df[~df[test_column].isin(allowed_values)]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found not allowed values")
        
    else:
        print("✅ Pass - found only allowed values")
    update_logbook(test_number, issues_df, 'testing no other values than specific occur in col', test_step)


def test_adding_WWW(start, test_val, end, test_number, test_step='', ):
    '''
    # sphinx-autodoc-skip
    '''
    result = end == start-test_val+2*test_val
    if result == True: 
        print("✅ passed the test! ")
        df = pd.DataFrame()
    else: 
        df = pd.DataFrame({'result': ['fail']})

    update_logbook(test_number, df, 
                   'testing addition of services / platform', test_step)

def test_adding_wseWWW_enw(start, test_val, end, test_number, test_step='', ):
    '''
    # sphinx-autodoc-skip
    '''
    result = end == start-test_val+4*test_val
    if result == True: 
        print("✅ passed the test! ")
        df = pd.DataFrame()
    else: 
        df = pd.DataFrame({'result': ['fail']})

    update_logbook(test_number, df, 
                   'testing addition of services / platform', test_step)


# test for above 1 where shouldnt'
def test_larger_val(df, column, test_number, test_step='', val=1):
    '''
    # sphinx-autodoc-skip
    '''
    # Check if any country/channel occurs more than once a week
    issues_df = df[df[column]>val]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found larger than 1 values")
        
    else:
        print("✅ Pass - No larger than 1 values")
    update_logbook(test_number, issues_df, 'testing the combination of columns for too large values', test_step)
    

def test_missing_hierarchy_levels(gam_info, df, test_number):
    '''
    # sphinx-autodoc-skip
    '''
    # Read the hierarchy from the Excel file
    hierarchy_df = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                                 sheet_name='ServiceID', 
                                 engine='openpyxl')[['ServiceID', 'ServiceName']]
    
    # filter to only services that are not in the dataset 
    # (enough to test parents because all services are in parents)
    issues_df = hierarchy_df[~hierarchy_df['ServiceID'].isin(df['ServiceID'])].drop_duplicates()
    
    update_logbook(test_number, issues_df, 'testing missing levels in hierarchy')
    
    if not issues_df.empty:
        print(f"❌ Test failed. Issues found and saved to '../test/{test_number}_issue_list.csv'.")
        return issues_df
    
    print("✅ All tests passed.")
    return issues_df

def test_merge_row_count(original_df, merged_df, test_number, test_step):
    '''
    # sphinx-autodoc-skip
    '''
    print('...testing if merge leads to more rows on the metric side')
    # Check if the number of rows in the merged DataFrame is less than or equal to the number of rows in the original DataFrame
    if len(merged_df) == len(original_df):
        print('✅ pass! :)')
        issue_list = pd.DataFrame()
    else:
        print('❌ fail :( ')
        common_columns = original_df.columns.intersection(merged_df.columns).tolist()
        issue_list = merged_df[merged_df.duplicated(subset=common_columns, keep=False)]
    
    # Log the result
    update_logbook(test_number, issue_list, 'testing merge row count', test_step)
    

# test for negative entries 
# sphinx-autodoc-skip
def test_negative_values(df, column, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    # Check if any country/channel occurs more than once a week
    issues_df = df[df[column] < 0]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found negative values")
        
    else:
        print("✅ Pass - No negative values")
    update_logbook(test_number, issues_df, 'testing the combination of columns for negative', test_step)


# test same values
# sphinx-autodoc-skip
def test_same_values(df1, df2, key, test_col, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    # Check if any country/channel occurs more than once a week
    test_calc = df1.merge(df2, on=key, how='left')
    issues_df = test_calc[test_calc[f'SUM {test_col}']!=test_calc[test_col]]

    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found some values")
        
    else:
        print("✅ Pass - No larger than 1 values")
    update_logbook(test_number, issues_df, 'testing that the summing between different steps is correct', test_step)


# sphinx-autodoc-skip
def test_outliers_general(df, numeric_columns, test_number, test_step='', threshold=3):
    """
    # sphinx-autodoc-skip
    
    Detect outliers in numeric columns using Z-score and log mean + allowed range.
    """
    issues_list = []
    for col in numeric_columns:
        if df[col].dtype.kind in 'biufc':  # numeric check
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - threshold * std, mean + threshold * std
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                for _, row in outliers.iterrows():
                    issues_list.append({
                        'Week': row['w/c'],
                        'Channel ID': row['Channel ID'],
                        'Column': col,
                        'Value': row[col],
                        'Mean': int(round(mean)),
                        'Allowed Range': f"[{lower:.2f}, {upper:.2f}]"
                    })
    issues_df = pd.DataFrame(issues_list)
    print(f"Test {test_number} {'❌ failed' if not issues_df.empty else '✅ passed'}: Outlier check.")
    update_logbook(test_number, issues_df, test='General outlier detection', test_step=test_step)


# sphinx-autodoc-skip
def test_outliers_vs_reference(df, reference_df, key_columns, numeric_columns, test_number, test_step='', tolerance=3):
    """
    # sphinx-autodoc-skip
    
    Detect outliers where actual value exceeds reference * (1 + tolerance).
    Only checks upper boundary and ignores reference points below 100k
    """
    
    merged = df.merge(reference_df, on=key_columns, suffixes=('', '_ref'))
    issues_list = []
    for col in numeric_columns:
        temp=merged[merged[f'{col}_ref'] > 100000]
        allowed_upper_col = f"{col}_allowed_upper"
        temp[allowed_upper_col] = temp[f"{col}_ref"] * (1 + tolerance)
        outliers = temp[temp[col] > temp[allowed_upper_col]]  # Upper-bound only
        if not outliers.empty:
            for _, row in outliers.iterrows():
                issues_list.append({
                    'Week': row['w/c'],
                    'Channel ID': row['Channel ID'],
                    'Column': col,
                    'Value': row[col],
                    'Reference': int(round(row[f"{col}_ref"])),
                    'Allowed Range': f"[0, {int(round(row[allowed_upper_col]))}]",
                    'Exceeded By': int(row[col] - row[allowed_upper_col])
                })
    issues_df = pd.DataFrame(issues_list)
    print(f"Test {test_number} {'❌ failed' if not issues_df.empty else '✅ passed'}: Upper-bound outlier check.")
    update_logbook(test_number, issues_df, test='Upper-bound outlier detection vs reference', test_step=test_step)

############################################################################################################
# VISUALISATION
def see_channel_week_heatmap(df, columns_to_visualize, week_col, id_col, 
                             id_name, bus_unit, file_path, gam_info):
    '''
    # sphinx-autodoc-skip
    '''
    for channel in df[id_col].unique():
        temp = df[df[id_col] == channel].sort_values([week_col], )
        # Define the columns to visualize
        
        warnings.filterwarnings("ignore", message="Glyph .* missing from font(s)")
    
        scaler = MinMaxScaler()
        temp[columns_to_visualize] = scaler.fit_transform(temp[columns_to_visualize])
        
        # Create a heatmap to visualize gaps in the dataset
        plt.figure(figsize=(10, 15))
        ax = sns.heatmap(temp[columns_to_visualize], cmap='viridis', 
                         yticklabels=pd.to_datetime(temp[week_col]).dt.strftime('%Y-%m-%d'))
        
        plt.xlabel('Metrics')
        plt.ylabel('weeks')
        plt.title(f"{temp[id_name].unique()[0]} - {temp[bus_unit].unique()[0]} - {file_path}")
        
        # Change the color bar legend to show 0 to max instead of 0 to 1
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(['0', 'max'])
    
        plt.savefig(f"../test/graphs/{file_path}_{channel}_{gam_info['file_timeinfo']}.pdf")
        plt.close()
        
def see_weekly_reach(gam_info, df, column, filename, date_col='w/c', with_nonNull_filter=False, store=False, subset=False):
    '''
    # sphinx-autodoc-skip
    '''
    df = df.sort_values('uniques', ascending=False)
    if with_nonNull_filter:
        print('excluding entries that have no 0 values')
        column_week_count = df.groupby([column, date_col])['uniques'].sum().reset_index().groupby(column)[date_col].count().reset_index()
        # create a list of all those column elements that occur less often than the number of weeks recorded for that year
        column_to_show = column_week_count[column_week_count[date_col] < gam_info['number_of_weeks']][column].to_list()
        df= df[df[column].isin(column_to_show)]
    
    if subset:

        number_of_hue = df[column].unique()
        markers = ['o', 's', 'D', '^', 'v',] #'<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']
        for i in range(len(number_of_hue)//5):
            # Create a temporary DataFrame for the current subset of 5 elements
            temp = df[df[column].isin(number_of_hue[i*5:(i+1)*5])]
            #temp['w/c'] += np.random.uniform(-0.5, 0.5, size=len(temp))
                
            # Create the line graph
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=temp, x=date_col, y='uniques', hue=column, 
                         marker= 'o',
                         errorbar=None, alpha=0.75, 
                        )
                         #marker='o', errorbar=None)
            
            # Set the title and labels
            plt.title(f'Reach for Each {column} Over Weeks')
            plt.xlabel('Week')
            plt.ylabel('Uniques')
            
            plt.yscale('log')
        
            # Show the plot
            if store:
                plt.savefig(f"../test/graphs/{filename}_{column}_{i}")
            else: 
                plt.show()
            plt.close()

    else:
        # Create the line graph
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=date_col, y='uniques', hue=column, 
                     marker= 'o',
                     errorbar=None, alpha=0.75, 
                    )
                     #marker='o', errorbar=None)
        
        # Set the title and labels
        plt.title(f'Reach for Each {column} Over Weeks')
        plt.xlabel('Week')
        plt.ylabel('Uniques')
        
        plt.yscale('log')
    
        # Show the plot
        if store:
            plt.savefig(f"../test/graphs/{filename}_{column}")
        else: 
            plt.show()
        plt.close()

############################################################################################################ 
# PODCAST RELATED

############################################################################################################ 
# FACEBOOK FACTORS
def test_engagement_logic(df):
    '''
    # sphinx-autodoc-skip
    '''
    test_fail = False
    for index, row in df.iterrows():
        
        if row['Engaged User + Autoplay'] < row['autoplay_viewer'] or row['Engaged User + Autoplay'] < row['Weekly Engaged Users']:
            print(f"Row {index}: FLAG - logic fails here!")
            test_fail = True
    if test_fail:
        print('❌ test failed!')
    else:
        print('✅ test passed!')

def youtube_test_input_files(test_number, folder_paths, main_path, week_tester, test_step=''):
    """
    # sphinx-autodoc-skip
    
    Function to test input files and check for issues.

    Parameters:
    test_number (str): The test number identifier.
    folder_paths (list): List of folder paths to check.
    main_path (str): Main path where folders are located.
    week_tester (pd.DataFrame): DataFrame containing week tester information.

    Returns:
    list: List of issues found during the test.
    """
    issues_list = []
    for folder in folder_paths:
        weeks_found = []
        for file_name in tqdm(os.listdir(main_path + folder)):
            if file_name.endswith('.zip'):    
                
                match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2}) (.+)', file_name)
                if match:
                    start_date = pd.to_datetime(match.group(1), format='%Y-%m-%d')
                    end_date   = pd.to_datetime(match.group(2), format='%Y-%m-%d')
                    
                    # Create a temporary DataFrame to perform checks
                    temp_df = pd.DataFrame({'w/c': [start_date], 'end_date': [end_date]})
        
                    if (temp_df['w/c'].dt.weekday != 0).any():
                        print('found another day than Monday!')
                        issues_list.append({'issue': 'start_date', 'file': folder + file_name})
        
                    if (temp_df['end_date'].dt.weekday != 0).any():
                        issues_list.append({'issue': 'end_date', 'file': folder + file_name})
        
                    if not ((temp_df['end_date'] - temp_df['w/c']).dt.days == 7).all():
                        issues_list.append({'issue': 'timeframe', 'file': folder + file_name})
                    
                    weeks_found.append(match.group(1))
                    
        week_tester_str = set(week_tester['w/c'].dt.strftime('%Y-%m-%d'))
        weeks_found_str = set(pd.to_datetime(weeks_found).strftime('%Y-%m-%d'))
        
        missing_weeks = week_tester_str - weeks_found_str

        if not missing_weeks:
            print("✅ All weeks are present in the dataset.")
        else:
            print("❌ Missing weeks:")
            print(missing_weeks)
            formatted_missing_weeks = [pd.to_datetime(week).strftime('%d-%m-%Y') for week in missing_weeks]
            issues_list.append({'issue': 'missing_weeks', 
                                'folder': folder, 
                                'missing weeks': ', '.join(formatted_missing_weeks)})
    
    
    update_logbook(test_number, pd.DataFrame(issues_list), 'all weeks in dataset', test_step)
    
        
def podcast_test_services_in_results(sql_results, podcast_details, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    # Extract the unique services from the podcast_details DataFrame
    expected_services = set(podcast_details['Service'].unique())
    
    # Extract the unique services from the SQL results DataFrame
    actual_services = set(sql_results['service'].unique())
    
    # Check if all expected services are in the actual services
    missing_services = expected_services - actual_services
    
    if not missing_services:
        print("✅ Pass - All services are listed in the SQL results.")
        issue_df = pd.DataFrame()
    else:
        print("❌ Fail - The following services are missing from the SQL results:")
        issue_df = pd.DataFrame(list(missing_services), columns=['Missing Services'])
        for service in missing_services:
            print(f"- {service}")
    update_logbook(test_number, issue_df, test='podcast_test_services_in_results', test_step='')


def podcast_check_unknown_services(sql_results, test_number, test_step=''):
    '''
    # sphinx-autodoc-skip
    '''
    # Check if 'Unknown' is present in the 'service' column
    unknown_services = sql_results[sql_results['service'] == 'Unknown']
    
    if unknown_services.empty:
        print("✅ Pass - No 'Unknown' services found in the SQL results.")
    else:
        print(f"❌ Fail - Found {len(unknown_services)} 'Unknown' services in the SQL results.")
        print(unknown_services)

    update_logbook(test_number, unknown_services, test='podcast_check_unknown_services', test_step='')

    
    
    
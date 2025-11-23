import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import re
from openpyxl import load_workbook
from openpyxl import Workbook

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.preprocessing import MinMaxScaler

########
import sys
from pathlib import Path

try:
    # Works in Python scripts
    helper_path = Path(__file__).resolve().parent.parent / "helper"
except NameError:
    # Works in Jupyter notebooks
    helper_path = Path().resolve().parent / "helper"

sys.path.insert(0, str(helper_path))

from config import gam_info
#########

# Test to check if all filter elements were returned
def test_filter_elements_returned(df, filter_elements, column_name, test_number, test_step=''):
    print(f"...testing {column_name}...")
    returned_elements = df[column_name].unique().tolist()
    missing_elements = set(filter_elements) - set(returned_elements)
    issues_df = pd.DataFrame(list(missing_elements), columns=[column_name])
    
    # Print the results
    if not issues_df.empty:
        print(f"❌ Test {test_number} failed: not all elements were retrieved")
    else:
        print(f"✅ Test {test_number} passed: everything found!")
    
    update_logbook(test_number=test_number, issues_list= issues_df, 
                   test_step=test_step, test='testing missing elements in columns')
    

def test_weeks_presence_per_account(key, id_column, main_data, week_lookup, test_number, test_step=''):
    """
    Check if all completed weeks from the week lookup are present for each account in the main dataset.

    Parameters:
    key (str): Column name for the week information.
    id_column (str): Column name for the account ID.
    main_data (pd.DataFrame): Main dataset containing account-week data.
    week_lookup (pd.DataFrame): DataFrame containing expected weeks.
    test_number (str): Unique test identifier.
    test_step (str): Optional description of the test step.

    Returns:
    pd.DataFrame: Missing account-week combinations if any.
    """
    # Ensure datetime consistency
    main_data[key] = pd.to_datetime(main_data[key])
    week_lookup[key] = pd.to_datetime(week_lookup[key])

    # Filter weeks to only include those fully completed (Monday last week or earlier)
    today = pd.Timestamp.today().normalize()
    last_monday = today - pd.Timedelta(days=today.weekday() + 7)  # Monday of last week
    completed_weeks = week_lookup[week_lookup[key] <= last_monday]

    # Create expected combinations using cartesian product
    expected_pairs = pd.MultiIndex.from_product(
        [main_data[id_column].unique(), completed_weeks[key].unique()],
        names=[id_column, key]
    )

    # Actual combinations from main data
    actual_pairs = pd.MultiIndex.from_frame(main_data[[id_column, key]])

    # Find missing pairs
    missing_pairs = expected_pairs.difference(actual_pairs)

    # Convert to DataFrame for logging
    if missing_pairs.empty:
        print(f"✅ Test {test_number} passed: All completed weeks present for each account.")
        missing_df = pd.DataFrame()
    else:
        print(f"❌ Test {test_number} failed: Missing completed weeks detected.")
        missing_df = pd.DataFrame(missing_pairs.to_frame(index=False))

    # Log results
    update_logbook(test_number, missing_df, test='Weeks presence per account', test_step=test_step)

def test_inner_join(df_left, df_right, key, test_number, test_step='', focus='both'):
    resulting_df = df_left[key].merge(df_right[key], on=key, how='outer', indicator=True, 
                                      suffixes=('_left', '_right'))
    
    # Initialize issue dataframes
    issue_df_left = pd.DataFrame()
    issue_df_right = pd.DataFrame()
    
    # Test to ensure no data is lost from df_left
    if (resulting_df[resulting_df['_merge'] == 'left_only'].shape[0] > 0) & (focus !='right'):
        issue_df_left = resulting_df[resulting_df['_merge'] == 'left_only']
    
    # Test to ensure no data is lost from df_right
    if (resulting_df[resulting_df['_merge'] == 'right_only'].shape[0] > 0) & (focus !='left'):
        issue_df_right = resulting_df[resulting_df['_merge'] == 'right_only']
    
    # Check if there are any issues with the join
    if issue_df_left.empty and issue_df_right.empty:
        print(f"✅ Inner join test {test_number} successful: No issues found.")
    else:
        
        print(f"Inner join test {test_number} failed: Issues found.")
        if not issue_df_left.empty:
            issue_df_left = issue_df_left.drop_duplicates()
            
            print(f"Issues with df_left (rows present in df_left but not in df_right)")
            
        if not issue_df_right.empty:
            issue_df_right = issue_df_right.drop_duplicates()
            
            print(f"Issues with df_right (rows present in df_right but not in df_left)")
            
    update_logbook(test_number, pd.concat([issue_df_left, issue_df_right]), 
                   test='testing inner join - dataloss between two tables', test_step=test_step)

# Test to check if the country percentage adds up to a 100% #former test_country_percentage
def test_percentage(df, groupby_columns, test_number, test_step, percentage_col='country_%'):

    test_df = df.copy()
    
    # Group by fb_page_name and fb_metric_end_time, and sum country_%
    test_df[percentage_col] = test_df[percentage_col] * 100
    test_df = test_df.groupby(groupby_columns)[percentage_col].sum().reset_index()
    test_df[percentage_col] = test_df[percentage_col].round(0)
    issues_df = test_df[test_df[percentage_col] != 100.0]

    update_logbook(test_number, issues_df, 'testing country percentage', test_step)

# test for duplicate entries 
def test_duplicates(df, columns, test_number, test_step=''):
    # Check if any country/channel occurs more than once a week
    issues_df = df.groupby(columns).size().reset_index(name='Count')
    issues_df = issues_df[issues_df['Count'] > 1]
    
    # Print the results
    if not issues_df.empty:
        print(f"❌ Test {test_number} failed: The following combinations occur more than once a week")
        
    else:
        print(f"✅ Test {test_number} passed: No combinations occurs more than once a week.")
    update_logbook(test_number, issues_df, 'testing the combination of columns for uniqueness', test_step)

def test_non_null_and_positive(df, numeric_columns=None, test_number='', test_step=''):
    """
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


def test_empty(df, test_number, test_step='', name='lookup'):
    """
    Test if a DataFrame is empty.
    Logs the result and updates the logbook.
    """
    issues_df = pd.DataFrame()

    if df.empty:
        print(f"❌ Test {test_number} failed: {name} DataFrame is empty.")
        issues_df = pd.DataFrame({'Issue': [f"{name} DataFrame is empty"]})
    else:
        print(f"✅ Test {test_number} passed: {name} DataFrame is not empty.")

    update_logbook(test_number, issues_df, test=f"Testing if {name} DataFrame is empty", test_step=test_step)

def test_missing(df, columns, test_number, test_step='', name='lookup'):
    """
    Test if there are missing values in specified columns of a DataFrame.
    Logs the result and updates the logbook.
    """
    # Count missing values in the specified columns
    missing_counts = df[columns].isnull().sum()
    issues_df = pd.DataFrame()

    if missing_counts.any():
        print(f"❌ Test {test_number} failed: Missing values detected in {name}.")
        # Prepare details for logbook
        missing_detail = {col: int(count) for col, count in missing_counts.items() if count > 0}
        issues_df = pd.DataFrame({'Issue': [f"Missing values: {missing_detail}"]})
    else:
        print(f"✅ Test {test_number} passed: No missing values in {name}.")

    update_logbook(test_number, issues_df, test=f"Testing missing values in {name}", test_step=test_step)
    
def test_lookup_files(df, id_columns, test_numbers, test_step):
    
    # returns entries
    test_empty(df, test_numbers[0], test_step=test_step, name='lookup')
    
    # unique keys 
    test_duplicates(df, id_columns, test_numbers[1], test_step=test_step)
    
    # missing values
    test_missing(df, id_columns, test_numbers[2], test_step=test_step, name='lookup')

#############################################################################################################
# test same values
def test_allowed_values(df, test_column, allowed_values, test_number, test_step=''):
    # Check if any country/channel occurs more than once a week
    issues_df = df[~df[test_column].isin(allowed_values)]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found not allowed values")
        
    else:
        print("✅ Pass - found only allowed values")
    update_logbook(test_number, issues_df, 'testing no other values than specific occur in col', test_step)


def test_hierarchy_reach(test_number, mode, gam_info, df, key, metric_col, test_step):
    """
    Test that the reach of each parent service is not smaller than any of its child services, but it is okay if it is smaller than the sum of its child services.

    Parameters:
    gam_info (dict): A dictionary containing information including the lookup file name.
    df (pd.DataFrame): A DataFrame containing reach data with columns 'ServiceID' and 'Reach'.
    test_number (str): The test number identifier.

    Returns:
    bool: True if the test passes, False otherwise.

    Example
    # Create a sample test DataFrame
    data = {
        'PlatformID': ['DPO', 'DPO', 'DPO', 'DPO', 'POD', 'POD', 'POD', 'POD'],
        'PlaceID': ['Country1', 'Country1', 'Country1', 'Country1', 'Country2', 'Country2', 'Country2', 'Country2'],
        'ServiceID': ['WSE', 'ARA', 'ANW', 'AX2', 'WSE', 'ANW', 'ARA', 'AX2'],
        'Reach': [100, 150, 80, 200, 300, 350, 250, 400]
    }
    
    # Convert the data to a DataFrame
    test_df = pd.DataFrame(data)
    
    # Display the DataFrame
    merged_df = test_hierarchy_reach(gam_info, test_df, "3_test")
    
    """
    if mode == 'Service':
        sheet_name = 'Service Hierarchy'
        col_name = 'ServiceID'

    if mode == 'Platform':
        sheet_name = 'Platform Hierarchy'
        col_name = 'PlatformID'
        
    # Read the hierarchy from the Excel file
    hierarchy_df = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name=sheet_name,
                                 engine='openpyxl')[['Parent', 'Child']].dropna()
    # filter to only services in dataset
    hierarchy_df = hierarchy_df[hierarchy_df['Parent'].isin(df[col_name]) & hierarchy_df['Child'].isin(df[col_name])]

    # Function to get all descendants of a parent using an iterative approach
    def get_descendants(df, parent):
        descendants = set()
        stack = [parent]
        while stack:
            current = stack.pop()
            children = df[df['Parent'] == current]['Child'].tolist()
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    stack.append(child)
        return descendants
    
    # Expand the parent-child combinations
    expanded_combinations = []
    for parent in hierarchy_df['Parent'].unique():
        descendants = get_descendants(hierarchy_df, parent)
        for descendant in descendants:
            expanded_combinations.append({'Parent': parent, 'Child': descendant})
    
    # Create a new DataFrame with expanded combinations
    hierarchy_df = pd.DataFrame(expanded_combinations)

    # add child reach platform and place
    merged_df = hierarchy_df.merge(df, left_on='Child', right_on=col_name, how='inner')
    merged_df = merged_df.rename(columns={metric_col: f'Child_{metric_col}'}).drop(columns=col_name)

    # add parent reach to service, platform and place
    merged_df = merged_df.merge(df, left_on=key+['Parent'], right_on=key+[col_name], how='inner')
    merged_df = merged_df.rename(columns={metric_col: f'Parent_{metric_col}'}).drop(columns=col_name)
    
    # Run the test
    issues_df = merged_df[merged_df[f'Child_{metric_col}'] > merged_df[f'Parent_{metric_col}']]
    issues_df['diff'] = issues_df[f'Child_{metric_col}'] - issues_df[f'Parent_{metric_col}']
    update_logbook(test_number, issues_df.sort_values('diff', ascending=False), 
                   'testing platform hierarchy reach', test_step)
    
    if not issues_df.empty:
        print("❌ Test failed. Issues found and saved to '../test/issues/'.")
        return issues_df
    
    print("✅ All tests passed.")
    return issues_df

def test_adding_WWW(start, test_val, end, test_number, test_step='', ):
    
    result = end == start-test_val+2*test_val
    if result == True: 
        print("✅ passed the test! ")
        df = pd.DataFrame()
    else: 
        df = pd.DataFrame({'result': ['fail']})

    update_logbook(test_number, df, 
                   'testing addition of services / platform', test_step)

def test_adding_wseWWW_enw(start, test_val, end, test_number, test_step='', ):
    
    result = end == start-test_val+4*test_val
    if result == True: 
        print("✅ passed the test! ")
        df = pd.DataFrame()
    else: 
        df = pd.DataFrame({'result': ['fail']})

    update_logbook(test_number, df, 
                   'testing addition of services / platform', test_step)



def test_join_rowCount(df_left, df_right, key, test_number, test_step=''):
    ''' this test ensures a join won't add additional rows to the reach data '''
    rowCount_before = df_left.shape[0]
    resulting_df = df_left[key].merge(df_right[key], on=key, how='left', indicator=True, 
                                      suffixes=('_left', '_right'))
    rowCount_after = resulting_df.shape[0]
    
    # Initialize issue dataframes
    issue_df = pd.DataFrame()
    
    # Test to ensure no data is lost from df_left
    if rowCount_before == rowCount_after:
        print(f"join - row count test {test_number} successful: No issues found.")   
    else:
        duplicated_keys = resulting_df[key][resulting_df.duplicated(key, keep=False)]
        issue_df = resulting_df[resulting_df[key].isin(duplicated_keys)]
            
    update_logbook(test_number, issue_df, 
                   test='testing row counts before/after join', test_step=test_step)
# test for above 1 where shouldnt'
def test_larger_val(df, column, test_number, test_step='', val=1):
    # Check if any country/channel occurs more than once a week
    issues_df = df[df[column]>val]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found larger than 1 values")
        
    else:
        print("✅ Pass - No larger than 1 values")
    update_logbook(test_number, issues_df, 'testing the combination of columns for too large values', test_step)
    
def test_missing_hierarchy_levels(gam_info, df, test_number):
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
def test_negative_values(df, column, test_number, test_step=''):
    # Check if any country/channel occurs more than once a week
    issues_df = df[df[column] < 0]
    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found negative values")
        
    else:
        print("✅ Pass - No negative values")
    update_logbook(test_number, issues_df, 'testing the combination of columns for negative', test_step)


# test same values
def test_same_values(df1, df2, key, test_col, test_number, test_step=''):
    # Check if any country/channel occurs more than once a week
    test_calc = df1.merge(df2, on=key, how='left')
    issues_df = test_calc[test_calc[f'SUM {test_col}']!=test_calc[test_col]]

    
    # Print the results
    if not issues_df.empty:
        print("❌ Fail - found some values")
        
    else:
        print("✅ Pass - No larger than 1 values")
    update_logbook(test_number, issues_df, 'testing that the summing between different steps is correct', test_step)


def test_weeks_presence(key, main_data, week_lookup, test_number, test_step=''):
    """
    Function to check if all weeks from the week lookup dataframe are present in the main dataset.

    Parameters:
    week_lookup (pd.DataFrame): DataFrame containing week lookup information.
    main_data (pd.DataFrame): DataFrame containing the main dataset.

    Returns:
    pd.DataFrame: DataFrame containing missing weeks if any, otherwise an empty DataFrame.
    """
    # make copies
    main_test_data = main_data.copy()
    week_lookup_test_data = week_lookup.copy()
    
    
    if key != 'Week Number':
        # ensure both are in the same dtype
        main_data[key] = pd.to_datetime(main_data[key])
        week_lookup[key] = pd.to_datetime(week_lookup[key])
    
    # Perform the join operation
    merged_data = pd.merge(main_test_data, week_lookup_test_data, on=key, how='inner')

    # Check for missing weeks
    missing_weeks = week_lookup_test_data[~week_lookup_test_data[key].isin(merged_data[key])]

    if missing_weeks.empty:
        print("✅ All weeks are present in the dataset.")
    else:
        print("❌ Missing weeks:")
        print(missing_weeks)
    update_logbook(test_number, missing_weeks, 'all weeks in dataset', test_step)

def test_outliers_general(df, numeric_columns, test_number, test_step='', threshold=3):
    """
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


def test_outliers_vs_reference(df, reference_df, key_columns, numeric_columns, test_number, test_step='', tolerance=3):
    """
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

'''def test_outliers_vs_reference(df, reference_df, key_columns, numeric_columns, test_number, test_step='', tolerance=0.5):
    """
    Detect deviations compared to reference data and log mean + allowed range.
    """
    merged = df.merge(reference_df, on=key_columns, suffixes=('', '_ref'))
    issues_list = []
    for col in numeric_columns:
        merged['diff_ratio'] = (merged[col] - merged[f"{col}_ref"]).abs() / merged[f"{col}_ref"].replace(0, 1)
        outliers = merged[merged['diff_ratio'] > tolerance]
        if not outliers.empty:
            for _, row in outliers.iterrows():
                #allowed_lower = row[f"{col}_ref"] * (1 - tolerance)
                allowed_lower = 0
                allowed_upper = row[f"{col}_ref"] * (1 + tolerance)
                issues_list.append({
                    'Week': row['w/c'],
                    'Channel ID': row['Channel ID'],
                    'Column': col,
                    'Value': row[col],
                    'Reference': row[f"{col}_ref"],
                    'Allowed Range': f"[{allowed_lower:.2f}, {allowed_upper:.2f}]"
                })
    issues_df = pd.DataFrame(issues_list)
    print(f"Test {test_number} {'❌ failed' if not issues_df.empty else '✅ passed'}: Outlier vs reference check.")
    update_logbook(test_number, issues_df, test='Outlier detection vs reference', test_step=test_step)
    '''
    
def update_logbook(test_number, issues_list, test='', test_step=''):
    """
    Function to update the logbook based on the presence of missing weeks.

    Parameters:
    test_number (str): The test number identifier.
    missing_weeks (pd.DataFrame): DataFrame containing missing weeks information.
    """
    print('...updating logbook...\n')
    logbook_path = "../test/test_logbook.xlsx"

    # ✅ If logbook does not exist, create it with headers
    if not os.path.exists(logbook_path):
        print("Logbook not found. Creating a new one...")
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        headers = ['test number', 'timestamp', 'pass/fail', 'check file', 'test', 'step']
        ws.append(headers)
        wb.save(logbook_path)

    # Load the specific sheet into a DataFrame
    logbook_df = pd.read_excel(logbook_path, sheet_name='Sheet1', engine='openpyxl')
    
    # Check if the test number exists in the logbook
    if test_number not in logbook_df['test number'].values:
        
        # Add a new row with the given test number
        new_row = pd.DataFrame({'test number': [test_number], 'timestamp': [''], 
                                'pass/fail': [''], 'check file': [''], 
                                'test': ['']})
        
        logbook_df = pd.concat([logbook_df, new_row], ignore_index=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logbook_df.loc[logbook_df['test number'] == test_number, 'timestamp'] = timestamp
    
    if test_step != '':
        logbook_df.loc[logbook_df['test number'] == test_number, 'test'] = test
    if test_step != '':
        logbook_df.loc[logbook_df['test number'] == test_number, 'step'] = test_step
        
    if not issues_list.empty:
        
        today_date = datetime.now().strftime('%Y-%m-%d')
        file_path = f"../test/issue_lists_{today_date}"
        file_name = f"/{test_number}_issue_list.csv"
        os.makedirs(file_path, exist_ok=True)

        issues_df = pd.DataFrame(issues_list)
        if 'Channel ID' in issues_df.columns:
            social_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype={'Channel ID': 'str'},
                                     sheet_name='Social Media Accounts new')[['Channel ID', 'Channel Name']]
            try:
                socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))
            except:
                pass
            issues_df.merge(social_accounts, on='Channel ID', how='left')
            if 'Channel Name' in issues_df.columns:
                # Reorder columns: Channel ID next to Channel Name
                cols = ['Channel ID', 'Channel Name'] + [c for c in issues_df.columns if c not in ['Channel ID', 'Channel Name']]
                issues_df = issues_df[cols]

        issues_df.to_csv(file_path+file_name, index=False)
        logbook_df.loc[logbook_df['test number'] == test_number, 'pass/fail'] = 'fail'
        logbook_df.loc[logbook_df['test number'] == test_number, 'check file'] = file_path+file_name
    else:
        logbook_df.loc[logbook_df['test number'] == test_number, 'pass/fail'] = 'pass'
        logbook_df.loc[logbook_df['test number'] == test_number, 'check file'] = 'no file created :)'
    

    # Write the updated DataFrame back to the specific sheet
    with pd.ExcelWriter(logbook_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        logbook_df.to_excel(writer, sheet_name='Sheet1', index=False)

############################################################################################################

def see_channel_week_heatmap(df, columns_to_visualize, week_col, id_col, 
                             id_name, bus_unit, file_path, gam_info):
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
# FACEBOOK FACTORS
def test_engagement_logic(df):
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



def site_test_unique_entries(df, column_name, test_number, test_step=''):
    # Check if all numbers in the specified column are unique
    unique_values = df[column_name].unique()
    total_values = df[column_name].count()
    
    duplicates = pd.DataFrame(columns=[column_name])
    if len(unique_values) == total_values:
        print(f"✅ Pass - All numbers in the column '{column_name}' are unique.")
    else:
        print(f"❌ Fail - There are duplicate numbers in the column '{column_name}'.")
        duplicates = df[column_name][df[column_name].duplicated()]
        print("Duplicate values:")
        print(duplicates)

    update_logbook(test_number, duplicates, test='site_test_unique_entries', test_step='')
    
        
def podcast_test_services_in_results(sql_results, podcast_details, test_number, test_step=''):
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
    # Check if 'Unknown' is present in the 'service' column
    unknown_services = sql_results[sql_results['service'] == 'Unknown']
    
    if unknown_services.empty:
        print("✅ Pass - No 'Unknown' services found in the SQL results.")
    else:
        print(f"❌ Fail - Found {len(unknown_services)} 'Unknown' services in the SQL results.")
        print(unknown_services)

    update_logbook(test_number, unknown_services, test='podcast_check_unknown_services', test_step='')

    
    
    
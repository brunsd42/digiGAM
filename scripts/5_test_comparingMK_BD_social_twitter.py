#!/usr/bin/env python
# coding: utf-8

# ## Status 
# - twitter business unit and aggregated services is currently calculated by using minnie's dataset (helper/tw_minnie_preBU.csv)
# - also minnie's weekly gnl dataset is missing week 51 & 52? 
# - 

# In[28]:


platformID = 'TWI'


# In[ ]:


import pandas as pd
pd.set_option('display.float_format', '{:.0f}'.format)


# In[29]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info
import functions


# In[30]:


# Load country mapping
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[['PlaceID', 'YouTube Codes']]
# Load country mapping
week_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')[['w/c', 'WeekNumber_finYear']]


# In[31]:


# Utility functions
def load_excel(path):
    return pd.read_excel(path, engine='openpyxl')

def load_csv(path):
    return pd.read_csv(path)

def standardize_country_codes(df, column='Country Code'):
    return df.replace({column: {'WLF': 'WFI', '* Total': 'Total'}})

def run_comparison(original_df, new_df, column_mapping, key_columns, method='integer', threshold=0.0001):
    if method == 'integer':
        return compare_dataframes_integer(original_df, new_df, column_mapping, key_columns)
    elif method == 'percentage':
        return compare_dataframes_percentage(original_df, new_df, column_mapping, key_columns, threshold)
    else:
        raise ValueError("Unknown comparison method")


# In[32]:


def compare_dataframes_integer(original_df, new_df, column_mapping, key_columns_new):
    """
    Compare two DataFrames and return rows that are missing or different.

    Parameters:
    - original_df: DataFrame from the original source
    - new_df: DataFrame from the new source
    - column_mapping: dict mapping original_df column names to new_df column names
    - key_columns_new: list of key columns using new_df naming

    Returns:
    - missing_from_new: rows in original_df not found in new_df
    - differing_rows: rows where key matches but mapped columns differ
    """

    # Rename original_df to match new_df column names
    original_df_renamed = original_df.rename(columns=column_mapping)

    # Ensure all required columns exist
    all_columns = list(column_mapping.values())
    original_subset = original_df_renamed[all_columns].copy()
    new_subset = new_df[all_columns].copy()

    # Round numeric columns to nearest integer
    for col in all_columns:
        if pd.api.types.is_numeric_dtype(original_subset[col]) and pd.api.types.is_numeric_dtype(new_subset[col]):
            original_subset[col] = original_subset[col].round(0).astype('Int64')
            new_subset[col] = new_subset[col].round(0).astype('Int64')
        
    # Merge to find differences
    merged = pd.merge(
        original_subset,
        new_subset,
        on=key_columns_new,
        how='outer',
        suffixes=('_orig', '_new'),
        indicator=True
    )

    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']

    # Differing rows: same keys but different values
    comparison_cols = [col for col in all_columns if col not in key_columns_new]
        
    differing_rows = merged[
        (merged['_merge'] == 'both') &
        merged[[f"{col}_orig" for col in comparison_cols]].ne(
            merged[[f"{col}_new" for col in comparison_cols]].values
        ).any(axis=1)
    ]

    return missing_from_new, differing_rows


# In[33]:


def compare_dataframes_percentage(original_df, new_df, column_mapping, key_columns_new, threshold=0.0001):
    """
    Compare two DataFrames and return rows that are missing or have percentage differences.

    Parameters:
    - original_df: DataFrame from the original source
    - new_df: DataFrame from the new source
    - column_mapping: dict mapping original_df column names to new_df column names
    - key_columns_new: list of key columns using new_df naming
    - threshold: minimum absolute difference to consider as significant

    Returns:
    - missing_from_new: rows in original_df not found in new_df
    - differing_rows: rows where key matches but mapped columns differ beyond threshold
    """

    # Rename original_df to match new_df column names
    original_df_renamed = original_df.rename(columns=column_mapping)

    # Ensure all required columns exist
    all_columns = list(column_mapping.values())
    original_subset = original_df_renamed[all_columns].copy()
    new_subset = new_df[all_columns].copy()

    # Merge to find differences
    merged = pd.merge(
        original_subset,
        new_subset,
        on=key_columns_new,
        how='outer',
        suffixes=('_orig', '_new'),
        indicator=True
    )

    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']

    # Compute differences
    comparison_cols = [col for col in all_columns if col not in key_columns_new]
    for col in comparison_cols:
        merged[f"{col}_diff"] = merged[f"{col}_new"] - merged[f"{col}_orig"]

    # Filter rows where any difference exceeds threshold
    diff_mask = merged['_merge'] == 'both'
    for col in comparison_cols:
        diff_mask &= merged[f"{col}_diff"].abs() > threshold

    differing_rows = merged[diff_mask]

    return missing_from_new, differing_rows


# In[ ]:





# In[34]:


# Dataset configuration
datasets = [
    
    {
        "name": "Engagements",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/final data/TW_GAM2025_REDSHIFT.xlsx",
        "new_path": f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.xlsx",
        "column_mapping": {
            "tw_account_id": "tw_account_id",
            "week_commencing": "w/c",
            "tweet_engagements": "tweet engagements"
        },
        "key_columns": ["tw_account_id", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "notes": "262 accounts in minnie's dataset - I suspect she got all accounts the BBC has "
    },
    
    {
        "name": "pre Business Unites",
        "original_path": "helper/tw_minnie_preBU.csv",
        "new_path": f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'TW Account ID': 'Channel ID', 
            'Twitter Engaged Users by Country': 'uv_by_country'
        },
        "key_columns": ["Channel ID", "w/c", 'PlaceID', 'ServiceID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "notes": ""
    },
    {
        "name": "GNL Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Twitter GNL.xlsx",
        "new_path": "../data/singlePlatform/output/weekly/GAM2025_WEEKLY_Twitter_GNLbyCountry.xlsx",
        "column_mapping": {
            "Service Code": "ServiceID",
            "Country Code": "PlaceID",
            "Twitter Engaged Users by Country": "Reach",
            "w/c": "w/c"
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
]
'''{
        "name": "Country",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2024/Social Media/data/Final Raw/Twitter Engagements inc country.xlsx",
        "new_path": f'../data/raw/{platformID}/stale_Twitter Engagements inc country.xlsx',
        "column_mapping": {
            "TW Account ID": "tw_account_id",
            "TW Service Code": 'ServiceID', 
            "Country": "PlaceID",
            "Week Number": "Week Number",
            "Engagement %": "country_%"
        },
        "key_columns": ["tw_account_id", "w/c", "PlaceID", "ServiceID"],
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "notes": "has older data in that isn't of the current GAM year"
    },'''


# In[41]:


datasets = [
    {
        "name": "GNL Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Twitter GNL.xlsx",
        "new_path": "../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_GNLbyCountry.xlsx",
        "column_mapping": {
            "Service Code": "ServiceID",
            "Country Code": "PlaceID",
            "Twitter Engaged Users by Country": "Reach",
            "w/c": "w/c"
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        },
        "notes": "the missing are all from the last two weeks. 
        I don't see these values in the alteryx workflow either so not sure where they come from in 
        the output file. 
        differences: looking at the biggest difference I fidn the same value from my calculation in
        alteryx but the excel sheet has a different number... "
    },
]
# Execute comparisons
for ds in datasets:
    print(f"\n--- Processing {ds['name']} ---")

    orig = functions.load_excel(ds["original_path"]) if ds["original_path"].endswith(".xlsx") else functions.load_csv(ds["original_path"])
    new  = functions.load_excel(ds["new_path"]) if ds["new_path"].endswith(".xlsx") else functions.load_csv(ds["new_path"])

    if ds["name"] == "Facebook Factors input":
        new['fb_page_id'] = new['fb_page_id'].astype(str)
        orig = orig.dropna(subset=['page_consumptions', 'page_video_complete_views_30s_autoplayed'],
                           how='any')
        
    # Special preprocessing for Country Percentage dataset
    if ds["name"] == "Country Percentage":
        
        # Rename 'Country' to 'YouTube Codes' in original data and merge with mapping
        orig = orig.rename(columns={'Country': 'YouTube Codes'})
        orig = orig.merge(country_map, on='YouTube Codes', how='left').drop(columns=['YouTube Codes'])

    if "Country Code" in orig.columns:
        orig = functions.standardize_country_codes(orig)
    if "Country Code" in new.columns:
        new = functions.standardize_country_codes(new)

    # Rename columns according to mapping
    orig = orig.rename(columns={k: v for k, v in ds["column_mapping"].items() if k in orig.columns})
    new  = new.rename(columns={k: v for k, v in ds["column_mapping"].items() if k in new.columns})

    # Special preprocessing for Country Percentage dataset
    if ds['preprocess']['week_mapping']:
        # add w/c using Week Number
        orig = orig.merge(week_map, left_on='Week Number', right_on='WeekNumber_finYear',
                                              how='left').drop(columns=['Week Number', 'WeekNumber_finYear'])

    # Ensure 'w/c' columns are datetime in both DataFrames
    if 'w/c' in orig.columns:
        orig['w/c'] = pd.to_datetime(orig['w/c'], errors='coerce')
    if 'w/c' in new.columns:
        new['w/c'] = pd.to_datetime(new['w/c'], errors='coerce')

    missing, different = functions.run_comparison(
        orig, new,
        ds["column_mapping"],
        ds["key_columns"],
        method=ds.get("method", "integer"),
        threshold=ds.get("threshold", 0.0001)
    )

    print("Rows missing from new:")
    display(missing)
    print("Rows with differences:")
    if 'Reach_new' in different.columns:
        if len(different) > 0:
            different['Reach_new'] = different['Reach_new'].fillna(0)
            different['diff'] = different['Reach_orig'] - different['Reach_new']
            display(different.sort_values('diff', ascending=False))
        else:
            display(different)
    else:
            display(different)


# In[40]:


orig[orig['w/c']=='2025-03-17']


# In[39]:


missing.shape


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[1]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info
import functions


# In[2]:


# Load country mapping
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[['PlaceID', 'YT-_FBE_codes']]
# Load country mapping
week_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')[['w/c', 'WeekNumber_finYear']]


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[8]:


# Dataset configuration
datasets = [
    {
        "name": "Facebook Engagement",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/final data/FB_GAM2025_REDSHIFT.xlsx",
        "new_path": "../data/raw/FBE/GAM2025_FBE_REDSHIFT.csv",
        "column_mapping": {
            'fb_page_id': 'fb_page_id', 
            'fb_metric_end_time': 'fb_metric_end_time',
            'page_consumptions_by_consumption_type': 'page_consumptions_by_consumption_type'
        },
        "key_columns": ["fb_page_id", "fb_metric_end_time"],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": '''only discrepancies have nan for values in metrics'''
    },
    {
        "name": "Facebook Country",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/final data/FB_GAM2025_REDSHIFT_COUNTRY.xlsx",
        "new_path": f"../data/raw/FBE/GAM2025_FBE_REDSHIFT_COUNTRY.csv",
        "column_mapping": {
            'fb_page_id': 'fb_page_id', 
            'fb_metric_id': 'fb_metric_id',
            'fb_metric_breakdown': 'YT-_FBE_codes',
            'fb_metric_end_time': 'week_ending',
            'country %': 'country_%',
        },
        "key_columns": ["fb_page_id", "fb_metric_id", "YT-_FBE_codes", "week_ending"],
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": """ yeeeah poifect! """
    },
    {
        "name": "Facebook Engagement & Country",
        "original_path": f"../test/alteryx_datasets/mk_FBE_uniqueViewer_country.csv",
        "new_path": f"../data/processed/FBE/GAM2025_FBE_uniqueViewer_country.csv",
        "column_mapping": {
            'Week Commencing': 'w/c', 
            'PLACEID1': 'PlaceID', 
            'FB Service Code': 'ServiceID', 
            'FB Page ID': 'Channel ID',
            'Engaged Users by Country': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'Channel ID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": """  
        163571453661989 & 2024-04-15: is also missing in minnie's raw dataset data\final data\FB_GAM2025_REDSHIFT.xlsx
        """
    }
]


# In[28]:


datasets = [

]
# Execute comparisons
for ds in datasets:
    # TODO - test currently doesn't catch additional things in my dataset that are not in minnie's 
    # e.g. I included Studios for UK / Youtube and Minnie did not - that did not show up here
    print(f"\n--- Processing {ds['name']} ---")

    orig = load_excel(ds["original_path"]) if ds["original_path"].endswith(".xlsx") else load_csv(ds["original_path"])
    new  = load_excel(ds["new_path"]) if ds["new_path"].endswith(".xlsx") else load_csv(ds["new_path"])

    # Special preprocessing for Country Percentage dataset
    if ds["name"] == "Country Percentage":
        # Rename 'Country' to 'YouTube Codes' in original data and merge with mapping
        orig = orig.rename(columns={'Country': 'YouTube Codes'})
        orig = orig.merge(country_map, on='YouTube Codes', how='left').drop(columns=['YouTube Codes'])

    if "Country Code" in orig.columns:
        orig = standardize_country_codes(orig)
    if "Country Code" in new.columns:
        new = standardize_country_codes(new)

    # Rename columns according to mapping
    orig = orig.rename(columns={k: v for k, v in ds["column_mapping"].items() if k in orig.columns})
    new  = new.rename(columns={k: v for k, v in ds["column_mapping"].items() if k in new.columns})

    # Special preprocessing for Country Percentage dataset
    if ds['preprocess']['week_mapping']:
        # add w/c using Week Number
        orig = orig.merge(week_map, left_on='Week Number', right_on='WeekNumber_finYear',
                                              how='left').drop(columns=['Week Number', 
                                                                        'WeekNumber_finYear'])

    # Ensure 'w/c' columns are datetime in both DataFrames
    col_names = ['w/c', 'fb_metric_end_time', 'week_ending']
    for date_col in col_names:
        if date_col in orig.columns:
            orig[date_col] = pd.to_datetime(orig[date_col], errors='coerce')
        if date_col in new.columns:
            new[date_col] = pd.to_datetime(new[date_col], errors='coerce')
    
    missing, different = run_comparison(
        orig, new,
        ds["column_mapping"],
        ds["key_columns"],
        method=ds.get("method", "integer"),
        threshold=ds.get("threshold", 0.0001)
    )

    print("Rows missing from new:")
    display(missing)
    print("Rows with differences:")
    if len(different) > 0:
        different['diff'] = different['Reach_orig'] - different['Reach_new']
        display(different.sort_values('diff', ascending=False))
    else:
        display(different)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from IPython.display import display


# In[2]:


import sys
from pathlib import Path

try:
    # Works in Python scripts
    helper_path = Path(__file__).resolve().parent.parent / "helper"
except NameError:
    # Works in Jupyter notebooks
    helper_path = Path().resolve().parent / "helper"

sys.path.insert(0, str(helper_path))

# Now import your modules 
from config import gam_info
import functions


# In[3]:


# Load country mapping
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)[['PlaceID', 'YT-_FBE_codes']]
# Load country mapping
week_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')[['w/c', 'WeekNumber_finYear']]


# In[4]:


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


# In[5]:


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

    
    # Differing rows: same keys but different values
    comparison_cols = [col for col in all_columns if col not in key_columns_new]
    
    merged = merged.dropna(subset=[f"{col}_orig" for col in comparison_cols])
    
    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']
    
    differing_rows = merged[
        (merged['_merge'] == 'both') &
        merged[[f"{col}_orig" for col in comparison_cols]].ne(
            merged[[f"{col}_new" for col in comparison_cols]].values
        ).any(axis=1)
    ]

    return missing_from_new, differing_rows


# In[6]:


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

    # Compute differences
    comparison_cols = [col for col in all_columns if col not in key_columns_new]

    merged = merged.dropna(subset=[f"{col}_orig" for col in comparison_cols])

    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']

    for col in comparison_cols:
        merged[f"{col}_diff"] = merged[f"{col}_new"] - merged[f"{col}_orig"]

    # Filter rows where any difference exceeds threshold
    diff_mask = merged['_merge'] == 'both'
    for col in comparison_cols:
        diff_mask &= merged[f"{col}_diff"].abs() > threshold

    differing_rows = merged[diff_mask]

    return missing_from_new, differing_rows


# In[7]:


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
#    {
#        "name": "Facebook Engagement & Country",
#        "original_path": f"../test/alteryx_datasets/mk_FBE_uniqueViewer_country.csv",
#        "new_path": f"../data/processed/FBE/GAM2025_FBE_uniqueViewer_country.csv",
#        "column_mapping": {
#            'Week Commencing': 'w/c', 
#            'PLACEID1': 'PlaceID', 
#           'FB Service Code': 'ServiceID', 
#            'FB Page ID': 'Channel ID',
#            'Engaged Users by Country': 'uv_by_country',
#        },
#        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'Channel ID'],
#        "method": "integer",
#        "preprocess": {
#            "standardize_country": False,
#            "week_mapping": False
#        },
#        "comment": """  
#        163571453661989 & 2024-04-15: is also missing in minnie's raw dataset data\final data\FB_GAM2025_REDSHIFT.xlsx
#        """
#    },
    {
        "name": "Facebook ALL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook ALL.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_ALLbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook ANW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook ANW.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_ANWbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook ANY Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook ANY.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_ANYbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook AX2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook AX2.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_AX2byCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook AXE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook AXE.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_AXEbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook EN2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook EN2 by country.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_EN2byCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook ENG Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook ENG by country.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_ENGbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook ENW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook ENW by country.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_ENWbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook FOA Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook FOA.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_FOAbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook GNL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook GNL.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_GNLbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook MA- Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook MA.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_MA-byCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook TOT Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook TOT.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_TOTbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook WOR Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook WOR.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_WORbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook WSE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook WSE.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_WSEbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
    {
        "name": "Facebook WSL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Facebook WSL.xlsx",
        "new_path": f"../data/singlePlatform/FBE/weekly/GAM2025_WEEKLY_FBE_WSLbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c', 
            'Country Code': 'PlaceID', 
            'Service Code': 'ServiceID', 
            'Platform': 'PlatformID',
            'Reach': 'Reach',
        },
        "key_columns": ['w/c', 'PlaceID', 'ServiceID', 'PlatformID'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": True
        },
        "comment": """  
        
        """
    },
]


# In[8]:


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
    if not different.empty:
        # Identify non-key columns (i.e., columns that are not part of the key_columns)
        key_cols = ds["key_columns"]
        metric_cols = [col.replace('_orig', '') for col in different.columns 
                       if col.endswith('_orig') and col.replace('_orig', '') not in key_cols]
    
        # Compute differences for each metric column
        for col in metric_cols:
            orig_col = f"{col}_orig"
            new_col = f"{col}_new"
            diff_col = f"{col}_diff"
            if orig_col in different.columns and new_col in different.columns:
                different[diff_col] = different[orig_col] - different[new_col]
    
        # Sort by the largest absolute difference in any metric column
        diff_cols = [f"{col}_diff" for col in metric_cols]
        sort_col = diff_cols[0] if diff_cols else None
        if sort_col:
            display(different.sort_values(by=sort_col, ascending=False))
        else:
            display(different)
    else:
        display(different)


# In[9]:


different['PlaceID'].unique()


# In[13]:


different[different['Reach_diff'] < -1]


# In[ ]:





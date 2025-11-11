#!/usr/bin/env python
# coding: utf-8

# In[44]:


platformID = 'INS'


# In[45]:


import pandas as pd
import numpy as np

from IPython.display import display


# In[46]:


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
from config_GAM2025 import gam_info
import functions


# In[47]:


# Load country mapping
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[['PlaceID', 'YT-_FBE_codes']]
# Load country mapping
week_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')[['w/c', 'WeekNumber_finYear']]
# social media accounts

dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
#socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))
channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# In[48]:


# Utility functions
def load_excel(path, sheet_name=0):
    return pd.read_excel(path, engine='openpyxl', sheet_name=sheet_name)

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


# In[49]:


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

    for col in key_columns_new:
        if pd.api.types.is_integer_dtype(original_subset[col]):
            original_subset[col] = original_subset[col].astype(str)
        if pd.api.types.is_integer_dtype(new_subset[col]):
            new_subset[col] = new_subset[col].astype(str)  
            
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


# In[50]:


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


# In[51]:


# Dataset configuration
datasets = [
    {
        "name": "Instagram Country - raw",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/final data/IG_GAM2025_REDSHIFT_geog.xlsx",
        "new_path": f"../data/raw/INS/{gam_info['file_timeinfo']}_INS_REDSHIFT_geog.csv",
        "column_mapping": 
        {
            'ig_user_id': 'ig_user_id', 
            'ig_user_name': 'ig_user_name',
            'ig_metric_breakdown': 'ig_metric_breakdown',
            'ig_metric_end_time': 'ig_metric_end_time',
            '% country': 'country_%',
        },
        "key_columns": ['ig_user_id', 'ig_user_name', 'ig_metric_breakdown', 'ig_metric_end_time'],
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": """ 
                    perfect!
                    """
},
    {
        "name": "Instagram Country - processed",
        "original_path": "../data/interim/temp_ig_country_processed.xlsx",
        "new_path": f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv",
        "column_mapping": 
        {
            'IG Platform Account ID': 'Channel ID', 
            #'IG Account Name': 'Channel Name',
            'fb_metric_breakdown': 'ig_metric_breakdown',
            'Week Commencing': 'w/c',
            'Country %': 'country_%',
        },
        "key_columns": ["Channel ID",  "ig_metric_breakdown", "w/c"], #'IG Account Name',
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": """ 
                    perfect! minnie's lookup has an error in the EastEnders Account ID 
                    which was corrected in my helper file
                    """
},
    {
        "name": "Instagram Engagement - raw",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/final data/IG_GAM2025_REDSHIFT.xlsx",
        "new_path": f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
        "column_mapping": {
            'IG Account ID': 'ig_user_id',
            'ig_metric_end_time': 'week_ending',
            #'ig_user_ig_id': 'ig_user_ig_id',
            'Weekly Reach': 'reach'
        },
        "key_columns": ['ig_user_id', 'week_ending'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": '''
                        perfect
        '''
    },
{
        "name": "Instagram Engagement - processed",
        "original_path": "../data/interim/temp_ig_engagement_processed.csv",
        "new_path": f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
        "column_mapping": {
            'IG Account ID': 'Channel ID',
            'Week Commencing': 'w/c',
            #'ig_user_ig_id': 'ig_user_ig_id',
            'Weekly Reach': 'reach'
        },
        "key_columns": ['Channel ID', 'w/c'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": '''
                        perfect
        '''
    },
    {
        "name": "Instagram combined",
        "original_path": "../data/interim/preBU_INS.csv",
        "new_path": f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv",
        "column_mapping": {
            'IG Account ID': 'Channel ID',
            'Week Commencing': 'w/c',
            'PLACEID1': 'PlaceID',
            'IG Reach by country': 'uv_by_country'
        },
        "key_columns": ['Channel ID', 'PlaceID', 'w/c'],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "comment": '''
                        can't explain discrepancy but I am pretty sure it is more likely to be an issue
                        in minnie's dataset. I stored csv's of the outputs and these are identical, 
                        however trying to use these as inputs in instagram_domi.yxmd failed. 
        '''
    },
    {
        "name": "Instagram ALL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - ALL by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_ALLbyCountry.xlsx",
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
        "name": "Instagram ANW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - ANW by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_ANWbyCountry.xlsx",
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
        "name": "Instagram ANY Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - ANY by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_ANYbyCountry.xlsx",
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
        "name": "Instagram AX2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - AX2 by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_AX2byCountry.xlsx",
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
        "name": "Instagram AXE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - AXE by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_AXEbyCountry.xlsx",
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
        "name": "Instagram EN2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - EN2 by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_EN2byCountry.xlsx",
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
        "name": "Instagram ENG Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - ENG by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_ENGbyCountry.xlsx",
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
        "name": "Instagram ENW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - ENW by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_ENWbyCountry.xlsx",
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
        "name": "Instagram FOA Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - FOA by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_FOAbyCountry.xlsx",
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
        "name": "Instagram GNL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - GNL by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_GNLbyCountry.xlsx",
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
        "name": "Instagram MA- Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - MA by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_MA-byCountry.xlsx",
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
        "name": "Instagram TOT Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - TOT by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_TOTbyCountry.xlsx",
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
        "name": "Instagram WOR Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - WOR by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_WORbyCountry.xlsx",
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
        "name": "Instagram WSE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - WSE by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_WSEbyCountry.xlsx",
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
        "name": "Instagram WSL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY Instagram - WSL by country.xlsx",
        "new_path": f"../data/singlePlatform/INS/weekly/GAM2025_WEEKLY_INS_WSLbyCountry.xlsx",
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


# In[55]:


# Execute comparisons
for ds in datasets:
    # TODO - test currently doesn't catch additional things in my dataset that are not in minnie's 
    # e.g. I included Studios for UK / Youtube and Minnie did not - that did not show up here
    print(f"\n--- Processing {ds['name']} ---")
    if ds["original_path"].endswith(".xlsx"):
        if 'original_sheetname' in ds.keys() :
            orig=load_excel(ds["original_path"], sheet_name=ds['original_sheetname'])
        else:
            orig=load_excel(ds["original_path"])
    else:
        orig = load_csv(ds["original_path"])
        
    new  = load_excel(ds["new_path"]) if ds["new_path"].endswith(".xlsx") else load_csv(ds["new_path"])
    
    print(orig.columns)
    # Special preprocessing for Country Percentage dataset
    if ds["name"] == "Country Percentage":
        # Rename 'Country' to 'YouTube Codes' in original data and merge with mapping
        orig = orig.rename(columns={'Country': 'YT-_FBE_codes'})
        orig = orig.merge(country_map, on='YT-_FBE_codes', how='left').drop(columns=['YT-_FBE_codes'])

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
    col_names = ['w/c', 'ig_metric_end_time', 'fb_metric_end_time', 'week_ending']
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
        numeric_cols = [col for col in different.columns if col.endswith('_orig') and pd.api.types.is_numeric_dtype(different[col])]
    
        for col in numeric_cols:
            base_col = col.replace('_orig', '')
            new_col = f"{base_col}_new"
            diff_col = f"{base_col}_diff"
            if new_col in different.columns:
                different[diff_col] = different[col] - different[new_col]
        
            display(different.sort_values(by=[col for col in different.columns if col.endswith('_diff')][0], ascending=False))
        else:
            display(different)


# In[ ]:





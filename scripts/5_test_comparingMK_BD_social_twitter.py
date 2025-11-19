#!/usr/bin/env python
# coding: utf-8

# ## Status 
# - twitter business unit and aggregated services is currently calculated by using minnie's dataset (helper/tw_minnie_preBU.csv)
# - also minnie's weekly gnl dataset is missing week 51 & 52? 
# - 

# In[1]:


platformID = 'TWI'


# In[2]:


from IPython.display import display

import pandas as pd
pd.set_option('display.float_format', '{:.0f}'.format)


# In[3]:


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


# In[4]:


# Load country mapping
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)[['PlaceID', 'YT-_FBE_codes']]
# Load country mapping
week_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')[['w/c', 'WeekNumber_finYear']]


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[11]:


datasets = [
    {
        "name": "Twitter ALL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter ALL.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_ALLbyCountry.xlsx",
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
        "name": "Twitter ANW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter ANW.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_ANWbyCountry.xlsx",
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
        "name": "Twitter ANY Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter ANY.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_ANYbyCountry.xlsx",
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
        "name": "Twitter AX2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter AX2.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_AX2byCountry.xlsx",
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
        "name": "Twitter AXE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter AXE.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_AXEbyCountry.xlsx",
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
        "name": "Twitter EN2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter EN2 by country.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_EN2byCountry.xlsx",
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
        "name": "Twitter ENG Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter ENG by country.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_ENGbyCountry.xlsx",
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
        "name": "Twitter ENW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter ENW by country.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_ENWbyCountry.xlsx",
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
        "name": "Twitter FOA Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter FOA.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_FOAbyCountry.xlsx",
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
        "name": "Twitter GNL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter GNL.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_GNLbyCountry.xlsx",
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
        "name": "Twitter MA- Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter MA.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_MA-byCountry.xlsx",
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
        "name": "Twitter TOT Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter TOT.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_TOTbyCountry.xlsx",
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
        "name": "Twitter WOR Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter WOR.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_WORbyCountry.xlsx",
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
        "name": "Twitter WSE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter WSE.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_WSEbyCountry.xlsx",
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
        "name": "Twitter WSL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly Twitter WSL.xlsx",
        "new_path": f"../data/singlePlatform/TWI/weekly/GAM2025_WEEKLY_TWI_WSLbyCountry.xlsx",
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


# In[12]:


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
    date_cols = ['w/c', 'week_ending']
    for col in date_cols:
        if col in orig.columns:
            orig[col] = pd.to_datetime(orig[col], errors='coerce')
        if col in new.columns:
            new[col] = pd.to_datetime(new[col], errors='coerce')

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





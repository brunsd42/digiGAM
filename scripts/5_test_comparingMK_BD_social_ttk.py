#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'TTK'


# In[2]:


from IPython.display import display

import pandas as pd


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
    for key in key_columns_new:
        # Convert both original and new key columns to string
        original_df[key] = original_df[key].astype(str).str.replace(r"\.0$", "", regex=True)
        new_df[key] = new_df[key].astype(str).str.replace(r"\.0$", "", regex=True)
    
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


def compare_dataframes_percentage(original_df, new_df, column_mapping, key_columns_new, 
                                  threshold=0.0001):
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
        #print(col)
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
        "name": "Country",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/TT country %.xlsx",
        "new_path": f"../data/processed/TTK/{gam_info['file_timeinfo']}_TTK_country.csv",
        "column_mapping": {
            "PlaceID (GAM Pivot)": "PlaceID",
            "Profile ID": "Channel ID",
            "% country": "percentage",
            },
        "key_columns": ["Channel ID", "PlaceID"],
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": True,
            "week_mapping": False
        },
        "notes": "perfect match"
    },
    {
        "name": "Unique Viewers",
        "original_path": "../data/interim/Tiktok raw data 10s test.xlsx",
        "new_path": f"../data/processed/TTK/GAM2025_TTK_views.csv",
        "column_mapping": {
            "w/c": "w/c",
            "Content ID": "content_id",
            "Tik Tok Service Code": "ServiceID",
            "Profile name": "Channel Name",
            "Final Video Views": "final_video_views"
        },
        "key_columns": ["content_id","Channel Name", "ServiceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": False
        },
        "notes": ""
    },
    {
        "name": "Unique Viewers + FB Views to Viewer",
        "original_path": "../test/alteryx_datasets/mk_ttk_viewerFB_join.csv",
        "new_path": f"../test/alteryx_datasets/db_ttk_viewerFB_join.csv",
        "column_mapping": {
            "w/c": "w/c",
            "Tik Tok Service Code": "ServiceID",
            "Profile ID": "Channel ID",
            "Views per viewer": "Views per viewer",
            #"Final Video Views": "Final Video Views"
        },
        "key_columns": ["Channel ID", "ServiceID", "w/c"],
        "method": "percentage",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": False
        },
        "notes": "Difference because Minnie has a mismatch in her join because of GNL vs * BBC GN"
    },
    {
        "name": "Unique Viewers + Country",
        "original_path": "../test/alteryx_datasets/mk_ttk_views_country.csv",
        "new_path": f"../test/alteryx_datasets/db_ttk_views_country.csv",
        "column_mapping": {
            "w/c": "w/c",
            "Tik Tok Service Code": "ServiceID",
            "Profile ID": "Channel ID",
            "PLACEID1": "PlaceID",
            "% country": "% country",
            },
        "key_columns": ["Channel ID", "ServiceID", "PlaceID", "w/c"],
        "method": "percentage",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": False
        },
        "notes": "Missing because Alteryx Workflow joins over name rather than ID"
    },
        {
        "name": "TikTok ALL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok ALL.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_ALLbyCountry.xlsx",
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
        "name": "TikTok ANW Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok ANW.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_ANWbyCountry.xlsx",
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
        "name": "TikTok ANY Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok ANY.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_ANYbyCountry.xlsx",
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
        "name": "TikTok AX2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok AX2.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_AX2byCountry.xlsx",
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
        "name": "TikTok AXE Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok AXE.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_AXEbyCountry.xlsx",
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
        "name": "TikTok EN2 Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok EN2 by country.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_EN2byCountry.xlsx",
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
        "name": "TikTok ENG Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok ENG by country.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_ENGbyCountry.xlsx",
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
        "name": "TikTok GNL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok GNL.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_GNLbyCountry.xlsx",
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
        "name": "TikTok MA- Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok MA.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_MA-byCountry.xlsx",
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
        "name": "TikTok TOT Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok TOT.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_TOTbyCountry.xlsx",
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
        "name": "TikTok WOR Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok WOR.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_WORbyCountry.xlsx",
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
        "name": "TikTok WSL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok WSL.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_WSLbyCountry.xlsx",
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


# In[29]:


datasets = [
        {
        "name": "TikTok GNL Platform",
        "original_path": f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/Weekly TikTok GNL.xlsx",
        "new_path": f"../data/singlePlatform/TTK/weekly/GAM2025_WEEKLY_TTK_GNLbyCountry.xlsx",
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
# Execute comparisons
for ds in datasets:
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
                                              how='left').drop(columns=['Week Number', 'WeekNumber_finYear'])

    # Ensure 'w/c' columns are datetime in both DataFrames
    if 'w/c' in orig.columns:
        orig['w/c'] = pd.to_datetime(orig['w/c'], errors='coerce')
    if 'w/c' in new.columns:
        new['w/c'] = pd.to_datetime(new['w/c'], errors='coerce')

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
    if 'Reach_new' in different.columns:
        if len(different) > 0:
            different['Reach_new'] = different['Reach_new'].fillna(0)
            different['diff'] = different['Reach_orig'] - different['Reach_new']
            display(different.sort_values('diff', ascending=False))
        else:
            display(different)
    else:
        display(different)


# In[33]:


new['PlaceID'].sort_values().unique()


# In[27]:


orig.shape


# In[28]:


missing.ServiceID.unique()


# In[ ]:





# In[ ]:





# In[ ]:





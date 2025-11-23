#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display

import pandas as pd


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
country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[['PlaceID', 'YT-_FBE_codes']]
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


# In[9]:


datasets = [
    {
        "name": "Unique Viewers",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/Final Raw/YouTube Unique Viewers.xlsx",
        "new_path": f"../data/processed/YT-/_{gam_info['file_timeinfo']}_uniqueViewer_withAds.csv",
        "column_mapping": {
            "Channel": "Channel ID",
            "YT Service Code": "ServiceID",
            "w/c": "w/c",
            "Unique viewers": "Unique viewers"
        },
        "key_columns": ["Channel ID", "ServiceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        }
    },
    {
        "name": "Country Percentage",
        "original_path": "../data/interim/yt_country_processed_interim.csv",
        "new_path": f"../data/processed/YT-/{gam_info['file_timeinfo']}_country_new.csv",
        "column_mapping": {
            "Channel": "Channel ID",
            "Country": "PlaceID",
            "Date": "w/c",
            "Country %": "country_%"
        },
        "key_columns": ["Channel ID", "w/c", "PlaceID"],
        "method": "percentage",
        "threshold": 0.0001,
        "preprocess": {
            "standardize_country": False,
            "week_mapping": False
        },
        "notes": '''
        there is one cahnnel missing that has NaN as country - checking countries there is for that 
        channel & date both UNK and NaN.
        '''
    },
    {
        "name": "GNL Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - GNL by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_GNLbyCountry.xlsx",
        "column_mapping": {
            "Service Code": "ServiceID",
            "Country Code": "PlaceID",
            "YouTube Engaged Reach": "Reach",
            "w/c": "w/c"
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "WSL Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - WSL by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_WSLbyCountry.xlsx",
        "column_mapping": {
            "Service Code": "ServiceID",
            "Country Code": "PlaceID",
            "YouTube Engaged Reach": "Reach",
            "w/c": "w/c"
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        },
        "notes": """
        missing are SER & SIN because I have placeID for them and with minnie they are missing
        """
    },
    {
        "name": "WOR Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - WOR by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_WORbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        },
        "notes": """
        missing are additional rows in minnie's dataset that have missing values for placeid and reach
        """
        
    },
    {
        "name": "WSE Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - WSE by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_WSEbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
        {
        "name": "MA- Weekly",
        "original_path": "../test/alteryx_datasets/mk_weekly_MA_YT.csv",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_MA-byCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        },
        "notes": """
            total should have been filtered out
        """
    },
    {
        "name": "FOA Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - FOA by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_FOAbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "AXE Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - AXE by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_AXEbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "AX2 Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - AX2 by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_AX2byCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "ANW Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - ANW by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_ANWbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "ANY Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - ANY by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_ANYbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "TOT Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - TOT by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_TOTbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "ALL Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - ALL by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_ALLbyCountry.xlsx",
        "column_mapping": {
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
            'w/c': 'w/c'
        },
        "key_columns": ["ServiceID", "PlaceID", "w/c"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "ENG Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - ENG by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_ENGbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c',
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
        },
        "key_columns": ["w/c", "ServiceID", "PlaceID"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "ENW Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - ENW by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_ENWbyCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c',
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
        },
        "key_columns": ["w/c", "ServiceID", "PlaceID"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
    {
        "name": "EN2 Weekly",
        "original_path": "../../../../Research Projects/GAM/Digital GAM/2025/Social Media/Output/Weekly/WEEKLY YouTube - EN2 by country.xlsx",
        "new_path": "../data/singlePlatform/YT-/weekly/GAM2025_WEEKLY_YT-_EN2byCountry.xlsx",
        "column_mapping": {
            'w/c': 'w/c',
            'Service Code': 'ServiceID', 
            'Country Code': 'PlaceID',
            'YouTube Engaged Reach': 'Reach',
        },
        "key_columns": ["w/c", "ServiceID", "PlaceID"],
        "method": "integer",
        "preprocess": {
            "standardize_country": True,
            "week_mapping": True
        }
    },
]


# In[10]:


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
    if len(different) > 0:
        different['diff'] = different['Reach_orig'] - different['Reach_new']
        display(different.sort_values('diff', ascending=False))
    else:
        display(different)


# In[ ]:





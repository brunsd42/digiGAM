#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[1]:


platformID = 'YT-'


# In[2]:


from IPython.display import display

import os
import zipfile

from tqdm import tqdm 
from datetime import datetime

import pandas as pd
pd.set_option('display.max_colwidth', None)

import numpy as np

import re

#import yxdb

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns 

import psycopg2


# ## import helper 

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

from functions import execute_sql_query
import test_functions


# In[4]:


# country
country_cols = ['PlaceID', 'YT-_FBE_codes']
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

### RUN TESTS
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_1country_0", f"{platformID}_1country_1", f"{platformID}_1country_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_1country_3", f"{platformID}_1country_4", f"{platformID}_1country_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_1country_6", f"{platformID}_1country_7", f"{platformID}_1country_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# # automated 

# In[5]:


sql_query = f"""
    SELECT
        week_commencing,
        channel_id,
        channel_name,
        country_code,
        views_country
    FROM
        central_insights.adverity_social_youtube_by_channel_geo
    WHERE
        week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
            AND
        channel_id IN ({formatted_channel_ids})
    ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_redshift_extract.csv"

#df = execute_sql_query(sql_query)    
#df.to_csv(file, index=False, na_rep='')

yt_views_raw = pd.read_csv(file, keep_default_na=False)
yt_views_raw['week_commencing'] = pd.to_datetime(yt_views_raw['week_commencing'])
yt_views_raw['country_code'] = yt_views_raw['country_code'].replace('', 'ZZ')
yt_views_raw = yt_views_raw.rename(columns = {'week_commencing': 'w/c',
                                              'channel_id': 'Channel ID',
                                              'channel_name': 'Channel Name',
                                              'country_code': 'YT-_FBE_codes'})

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(yt_views_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"9_{platformID}_country",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='Channel ID',
                                               main_data=yt_views_raw,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"10_{platformID}_country",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(yt_views_raw, 
                           numeric_columns=['views_country'], 
                           test_number=f"11_{platformID}_country",
                           test_step='Check no missing values in page fans column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(yt_views_raw, ['Channel ID', 'w/c', 'YT-_FBE_codes'], 
                               test_number=f"12_{platformID}_country",
                               test_step='Check no duplicates from redshift returned')


# In[6]:


# add PlaceID
cols = ['PlaceID', 'YT-_FBE_codes']
yt_views = yt_views_raw.merge(country_codes[cols],
                              on=['YT-_FBE_codes'],
                              how='left').drop(columns=['YT-_FBE_codes'])
test_functions.test_inner_join(yt_views_raw, 
                               country_codes[cols], 
                               ['YT-_FBE_codes'], 
                               f"13_{platformID}_country",
                               test_step='adding country codes GAM', 
                               focus='left')



# In[7]:


# Group by the specified columns and sum the yt_metric_value
yt_views_global = yt_views.groupby([ 'Channel ID','w/c']).agg({'views_country': 'sum'}).reset_index()
yt_views_global = yt_views_global.rename(columns={'views_country': 'total_views_country'})
#display(grouped_df_allCountries.sample())

yt_country = yt_views.merge(yt_views_global,
                                    on=['Channel ID', 'w/c'],
                                    how='inner')
test_functions.test_inner_join(yt_views, 
                               yt_views_global, 
                               ['Channel ID', 'w/c'],
                               f"14_{platformID}_country",
                               test_step='combining country and global views', 
                               focus='both')


yt_country['country_%'] = (yt_country['views_country'] / yt_country['total_views_country'])

test_functions.test_percentage(yt_country,  
                               ['Channel ID', 'w/c'], 
                               f"15_{platformID}_country",
                               test_step = 'calculating % country')
yt_country.sample()


# # manual

# # store

# In[8]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
yt_country[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_REDSHIFT_COUNTRY.csv", 
                         index=None, na_rep='')


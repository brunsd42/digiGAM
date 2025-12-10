#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'FBE'


# ## import libraries

# In[2]:


import os
import zipfile

from tqdm import tqdm 
from datetime import datetime

import pandas as pd
pd.set_option('display.max_colwidth', None)

import numpy as np

import re

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
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID',
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

### RUN TESTS
test_functions.test_lookup_files(country_codes, ['PlaceID'], [f"{platformID}_3_0", f"{platformID}_3_1", f"{platformID}_3_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3_3", f"{platformID}_3_4", f"{platformID}_3_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_3_6", f"{platformID}_3_7", f"{platformID}_3_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")



# # Unique Viewers

# In[5]:


facebook_engagements_reach = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",)
facebook_engagements_reach['w/c'] = pd.to_datetime(facebook_engagements_reach['w/c'])
facebook_engagements_reach['Channel ID'] = facebook_engagements_reach['Channel ID'].apply(lambda x: str(int(x)))


# # Country

# In[6]:


country_raw = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv")
country_raw['w/c'] = pd.to_datetime(country_raw['w/c'])
country_raw['Channel ID'] = country_raw['Channel ID'].apply(lambda x: str(int(x)))


cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
country_df = country_raw[cols]
country_df.head()


# In[7]:


# get 
country_lastYear = pd.read_excel(f"../data/stale/2024_FBE_Engagements_Country.xlsx", 
                                 sheet_name='Weekly FB Country USE')
country_lastYear = country_lastYear.rename(columns={
    'FB Page ID': 'Channel ID', 
    'Country Code': 'YT-_FBE_codes', 
    'country %': 'country_%'
})
country_lastYear['Channel ID'] = country_lastYear['Channel ID'].apply(lambda x: str(int(x)))

country_lastYear = country_lastYear.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], on=['YT-_FBE_codes'], how='left', 
                                          indicator=True)
print(country_lastYear._merge.value_counts())
country_lastYear['PlaceID'] = country_lastYear['PlaceID'].fillna('UNK')
avg_country_df = country_lastYear.groupby(['Channel ID', 'PlaceID'])['country_%'].mean().reset_index()


# In[8]:


reach_df_raw = facebook_engagements_reach.merge(country_df, on=['Channel ID', 'w/c'], how='outer', 
                                            indicator=True)

reach_df_left = reach_df_raw[reach_df_raw['_merge'] == 'left_only'].drop(columns=['_merge'])
reach_df_inner = reach_df_raw[reach_df_raw['_merge'] == 'both'].drop(columns=['_merge'])

reach_df_avg = reach_df_left[facebook_engagements_reach.columns].merge(avg_country_df, 
                                    on=['Channel ID'], how='left', indicator=True)

reach_df = pd.concat([reach_df_inner, reach_df_avg])


# In[9]:


metric_col = ['country_%', 'engaged_reach']
for col in metric_col:
    reach_df[col] = reach_df[col].fillna(0)
    
reach_df['uv_by_country'] = reach_df['country_%'] * reach_df['engaged_reach']
# TODO investigate why there should be duplicates here
reach_df = reach_df.drop_duplicates()

print(reach_df.shape)
reach_df = reach_df.dropna(subset='uv_by_country')
print(reach_df.shape)

cols = ['ServiceID', 'Channel ID', 'w/c', 'PlaceID', 'uv_by_country']
reach_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv", 
                     index=None)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[50]:


platformID = 'FBE'


# ## import libraries

# In[51]:


import os
import zipfile

from tqdm import tqdm 
from datetime import datetime

import pandas as pd
pd.set_option('display.max_colwidth', None)

import numpy as np

import re

import yxdb

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns 

import psycopg2


# ## import helper 

# In[52]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[53]:


gam_info['lookup_file']


# In[54]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID',
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.dtypes


# # Unique Viewers

# In[55]:


facebook_engagements_reach = pd.read_csv(f"../data/processed/FBE/{gam_info['file_timeinfo']}_FBE_REDSHIFT.csv",)
facebook_engagements_reach['w/c'] = pd.to_datetime(facebook_engagements_reach['w/c'])
facebook_engagements_reach['Channel ID'] = facebook_engagements_reach['Channel ID'].apply(lambda x: str(int(x)))


# # Country

# In[56]:


country_raw = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv")
country_raw['w/c'] = pd.to_datetime(country_raw['w/c'])
country_raw['fb_page_id'] = country_raw['fb_page_id'].apply(lambda x: str(int(x)))


cols = ['fb_page_id', 'fb_page_name', 'w/c', 'PlaceID', 'country_%']
country_df = country_raw[cols].rename(columns={"fb_page_id": "Channel ID"})
country_df.head()


# In[57]:


'''# test for duplicates in ['fb_page_id', 'w/c', 'Country Code', 'country_%']
country_df = country_raw.merge(country_codes.rename(columns={'YouTube Codes': 'Country Code'})[['Country Code', 'PlaceID', gam_info['population_column]]],
                               on='Country Code', how='left', indicator=True)
# test for clean merge
#print(country_df._merge.value_counts())
country_df = country_df.drop(columns=['_merge'])
country_df.dtypes
'''
avg_country_df = country_df.groupby(['Channel ID', 'PlaceID'])['country_%'].mean().reset_index()
avg_country_df.head()


# In[58]:


reach_df_raw = facebook_engagements_reach.merge(country_df, on=['Channel ID', 'w/c'], how='outer', 
                                            indicator=True)

reach_df_left = reach_df_raw[reach_df_raw['_merge'] == 'left_only'].drop(columns=['_merge'])
reach_df_inner = reach_df_raw[reach_df_raw['_merge'] == 'both'].drop(columns=['_merge'])

reach_df_avg = reach_df_left[facebook_engagements_reach.columns].merge(avg_country_df, 
                                    on=['Channel ID'], how='left', indicator=True)

reach_df = pd.concat([reach_df_inner, reach_df_avg])
reach_df.head()


# In[61]:


reach_df['uv_by_country'] = reach_df['country_%'] * reach_df['Engaged Reach']
reach_df = reach_df.drop_duplicates()

cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
reach_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv", 
                     index=None)


# In[ ]:





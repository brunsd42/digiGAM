#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[67]:


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

# In[68]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[69]:


gam_info['lookup_file']


# In[70]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['Platform'] == 'Youtube')
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.sample()


# # Unique Viewers

# In[71]:


uniqueViewer_df = pd.read_csv(f"../data/processed/Youtube/{gam_info['file_timeinfo']}_uniqueViewer.csv")
uniqueViewer_df.sample()


# In[84]:


uniqueViewer_df[(uniqueViewer_df['ServiceID'] == 'SER') & 
                (uniqueViewer_df['w/c'] == '2024-04-01')]


# In[93]:


temp = country_df.merge(socialmedia_accounts[['Channel ID', 'Service']], on='Channel ID', how='left')
uniqueViewer_df[(uniqueViewer_df['w/c'] == '2024-07-01') & (uniqueViewer_df['ServiceID'] == 'MA-')]['Unique viewers'].sum()


# # Country

# In[72]:


country_df = pd.read_csv(f"../data/processed/Youtube/{gam_info['file_timeinfo']}_country.csv")
country_df.sample()


# In[90]:


temp = country_df.merge(socialmedia_accounts[['Channel ID', 'Service']], on='Channel ID', how='left')
temp[(temp['PlaceID'] == 'INO')  & (temp['w/c'] == '2024-07-01') & (temp['Service'] == 'Media Action')]


# # combine UV & country

# In[73]:


yt_uv_country = uniqueViewer_df.merge(country_df, 
                            on=['Channel ID', 'w/c'], 
                            how = 'outer', indicator=True)

################################### Testing ################################### 
test_step = 'merging uv & country'

test_functions.test_inner_join(uniqueViewer_df, country_df, ['Channel ID', 'w/c'], '1_YT_18', test_step)

################################### Testing ################################### 

print(yt_uv_country._merge.value_counts())


# In[74]:


#yt_uv_country.to_csv('temp_yt_uvCountry.csv', index=None)


# In[75]:


yt_uv_country = yt_uv_country[yt_uv_country['_merge'] == 'both'].drop(columns=['_merge'])
yt_uv_country['uv_by_country'] = yt_uv_country['country_%'] * yt_uv_country['Unique viewers']
yt_uv_country.sample()


# In[76]:


################################### Testing ################################### 
# all weeks 
# all weeks per channel
# all weeks per service
# all weeks per country

# duplicates

# test unique viewer is larger than unique vieewr per country 
# test total of unique vieewr per country == unique viewer
# test country sum == 1

# test allowed values - placeID
# test allowed values - ServiceID

################################### Testing ################################### 


# country tests
# - [ ] check only one entry per country & week & channel
# - [ ] check that sum of unique views per country == unique views gathered from yt clickedicklick
# - [ ] check country sum==1 (groupby channel & week == 1)
# - [ ] county the number of weeks we have for every channel counntry combination
# - [ ] check that no channel name is empty

# In[ ]:





# In[77]:


cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', ]
yt_uv_country[cols].to_csv(f"../data/processed/YouTube/{gam_info['file_timeinfo']}_uniqueViewer_country.csv", 
                     index=None)
'''yt_uv_country.to_csv(f"../data/singlePlatform/input/YouTube/{gam_info['file_timeinfo']}_metric_country.csv", 
                     index=None)'''


# In[80]:


yt_uv_country[(yt_uv_country['ServiceID'] == 'SER') & 
        (yt_uv_country['w/c'] == '2024-04-01') ].PlaceID.unique()


# In[ ]:





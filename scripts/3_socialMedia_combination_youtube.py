#!/usr/bin/env python
# coding: utf-8

# In[60]:


platformID = 'YT-'


# ## import libraries

# In[61]:


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

# In[62]:


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

from functions import execute_sql_query
import test_functions


# In[63]:


gam_info['lookup_file']


# In[64]:


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

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.sample()


# # Unique Viewers

# In[65]:


uniqueViewer_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer.csv")
uniqueViewer_df.sample()


# In[66]:


uniqueViewer_df[(uniqueViewer_df['ServiceID'] == 'SER') & 
                (uniqueViewer_df['w/c'] == '2024-04-01')]


# # Country

# In[67]:


country_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_country_new.csv")
country_df.sample()


# In[68]:


temp = country_df.merge(socialmedia_accounts[['Channel ID', 'Service']], on='Channel ID', how='left')
temp[(temp['PlaceID'] == 'INO')  & (temp['w/c'] == '2024-07-01') & (temp['Service'] == 'Media Action')]


# # combine UV & country

# In[69]:


yt_uv_country = uniqueViewer_df.merge(country_df, 
                            on=['Channel ID', 'w/c'], 
                            how = 'outer', indicator=True)

################################### Testing ################################### 
test_step = 'merging uv & country'

test_functions.test_inner_join(uniqueViewer_df, country_df, ['Channel ID', 'w/c'], '1_YT_18', test_step)

################################### Testing ################################### 

print(yt_uv_country._merge.value_counts())


# In[70]:


#yt_uv_country.to_csv('temp_yt_uvCountry.csv', index=None)


# In[71]:


yt_uv_country = yt_uv_country[yt_uv_country['_merge'] == 'both'].drop(columns=['_merge'])
yt_uv_country['uv_by_country'] = yt_uv_country['country_%'] * yt_uv_country['Unique viewers']
yt_uv_country.sample()


# In[72]:


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

# In[73]:


cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', ]
yt_uv_country[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv", 
                     index=None)
'''yt_uv_country.to_csv(f"../data/singlePlatform/input/YouTube/{gam_info['file_timeinfo']}_metric_country.csv", 
                     index=None)'''


# In[ ]:





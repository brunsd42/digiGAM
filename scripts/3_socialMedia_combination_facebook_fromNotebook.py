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

from functions import calculate_rolling_avg_country_split
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


# In[6]:


facebook_engagements_reach.head()


# # Country

# In[7]:


country_raw = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv")
country_raw['w/c'] = pd.to_datetime(country_raw['w/c'])
country_raw['Channel ID'] = country_raw['Channel ID'].apply(lambda x: str(int(x)))


cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
country_df = country_raw[cols]
country_df['PlatformID'] = platformID
country_df.head()


# In[8]:


avg_country_df = calculate_rolling_avg_country_split(country_df, metric_col='country_%')


# # combine engagements & country

# In[9]:


reach_df_raw = facebook_engagements_reach.merge(country_df, on=['Channel ID', 'w/c'], how='outer', 
                                            indicator=True)

reach_df_left = reach_df_raw[reach_df_raw['_merge'] == 'left_only'].drop(columns=['_merge'])
reach_df_inner = reach_df_raw[reach_df_raw['_merge'] == 'both'].drop(columns=['_merge'])

reach_df_avg = reach_df_left[facebook_engagements_reach.columns].merge(avg_country_df, 
                                    on=['Channel ID', 'w/c'], how='left', indicator=True)

reach_df = pd.concat([reach_df_inner, reach_df_avg])


# In[10]:


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


# In[11]:


reach_df.head()


# In[12]:


reach_df[reach_df['ServiceID'].isin(['BNI', 'BNO', 'GNL'])]


# In[13]:


reach_df[(reach_df['w/c'] == '2025-12-01')  & 
    (reach_df['Channel ID'].isin(['630866223444617']))]


# In[14]:


avg_country_df[avg_country_df['Channel ID'].isin(['630866223444617'])]


# In[ ]:





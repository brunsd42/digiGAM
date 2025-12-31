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

from functions import lookup_loader
import test_functions


# In[4]:


lookup = lookup_loader(gam_info, platformID, '3',
                       with_country=True)
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']
country_codes = lookup['country_codes']


# # Unique Viewers

# In[5]:


facebook_engagements_reach = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",)
facebook_engagements_reach['w/c'] = pd.to_datetime(facebook_engagements_reach['w/c'])



# In[6]:


facebook_engagements_reach.sort_values(['w/c'])['w/c'].unique()


# # Country

# In[7]:


country_raw = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv")
country_raw['w/c'] = pd.to_datetime(country_raw['w/c'])

cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
country_df = country_raw[cols]
country_df['PlatformID'] = platformID
country_df.head()


# In[8]:


country_df.sort_values(['w/c'])['w/c'].unique()


# # combine engagements & country

# In[9]:


country_df[(country_df['Channel ID'] == 'FBE630866223444617') 
    #& (country_df['w/c'].isin(['2025-06-23', '2025-06-30']))
    ]


# In[10]:


reach_df_raw = facebook_engagements_reach.merge(country_df, on=['Channel ID', 'w/c'], how='outer', 
                                            indicator=True)
print(reach_df_raw._merge.value_counts())
reach_df_left = reach_df_raw[reach_df_raw['_merge'] == 'left_only'].drop(columns=['_merge'])
reach_df = reach_df_raw[reach_df_raw['_merge'] == 'both'].drop(columns=['_merge'])

'''
reach_df_inner = reach_df_raw[reach_df_raw['_merge'] == 'both'].drop(columns=['_merge'])
reach_df_avg = reach_df_left[facebook_engagements_reach.columns].merge(avg_country_df, 
                                    on=['Channel ID', 'w/c'], how='left', indicator=True)

reach_df = pd.concat([reach_df_inner, reach_df_avg])'''


# In[11]:


metric_col = ['country_%', 'engaged_reach']
for col in metric_col:
    reach_df[col] = reach_df[col].fillna(0)
    
reach_df['uv_by_country'] = reach_df['country_%'] * reach_df['engaged_reach']
reach_df = reach_df[reach_df['uv_by_country'] > 0]
# TODO investigate why there should be duplicates here
reach_df = reach_df.drop_duplicates()

print(reach_df.shape)
reach_df = reach_df.dropna(subset='uv_by_country')
print(reach_df.shape)


# In[12]:


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col='Channel ID',
                                               main_data=reach_df,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts,
                                               test_number=f"{platformID}_3_09",
                                               test_step="Check all weeks present for each account")


# In[13]:


reach_df.sort_values(['w/c'])['w/c'].unique()


# In[14]:


cols = ['ServiceID', 'Channel ID', 'w/c', 'PlaceID', 'uv_by_country']
reach_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv", 
                     index=None)


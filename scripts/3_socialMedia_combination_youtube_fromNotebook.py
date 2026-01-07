#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'YT-'


# ## import libraries

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


gam_info['lookup_file']


# In[5]:


# country
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
test_functions.test_lookup_files(country_codes, ['PlaceID'], [f"{platformID}_3_0", f"{platformID}_3_1", f"{platformID}_3_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3_3", f"{platformID}_3_4", f"{platformID}_3_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_3_6", f"{platformID}_3_7", f"{platformID}_3_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")



# # Unique Viewers

# In[6]:


uniqueViewer_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer.csv")
uniqueViewer_df.sample()


# In[7]:


uniqueViewer_df['w/c'].max()


# # Country

# In[8]:


country_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_REDSHIFT_COUNTRY.csv")
country_df.sample()
country_df['w/c'].max()


# # combine UV & country

# In[9]:


yt_uv_country = uniqueViewer_df.merge(country_df, 
                            on=['Channel ID', 'w/c'], 
                            how = 'outer', indicator=True)

################################### Testing ################################### 
test_step = 'merging uv & country'

test_functions.test_inner_join(uniqueViewer_df, country_df, 
                               ['Channel ID', 'w/c'], 
                               f"{platformID}_3_09", test_step)

################################### Testing ################################### 

print(yt_uv_country._merge.value_counts())


# In[10]:


yt_uv_country = yt_uv_country[yt_uv_country['_merge'] == 'both'].drop(columns=['_merge'])
yt_uv_country['uv_by_country'] = yt_uv_country['country_%'] * yt_uv_country['Unique viewers']
yt_uv_country.sample()


# In[11]:


print(yt_uv_country.shape)
yt_uv_country = yt_uv_country.dropna(subset='uv_by_country')
print(yt_uv_country.shape)

cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', ]
yt_uv_country[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv", 
                     index=None)


# In[ ]:





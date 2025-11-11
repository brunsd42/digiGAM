#!/usr/bin/env python
# coding: utf-8

# country isn't updated anymore so we are using an old year 

# In[10]:


platformID = 'TWI'


# In[11]:


from datetime import datetime
import pandas as pd

import psycopg2


# ## import helper

# In[12]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[13]:


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

socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == {platformID}]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# # ingestion

# # country

# In[17]:


# takes forever to load
tw_country_df = pd.read_excel(f'../data/raw/{platformID}/stale_Twitter Engagements inc country.xlsx')

# ensure the accounts are strings (TW & FB)
tw_country_df['TW Account ID'] = tw_country_df['TW Account ID'].astype(str)
tw_country_df['TW Linked FB account'] = tw_country_df['TW Linked FB account'].astype(str).str.split('.').str[0]

tw_country_df = tw_country_df.rename(columns={"TW Account ID": "tw_account_id"})
tw_country_df = tw_country_df.rename(columns={'Week Number': 'WeekNumber_finYear'})

tw_country_df.sample()


# In[18]:


# test accounts 
column_name = 'tw_account_id'
test_functions.test_filter_elements_returned(tw_country_df, channel_ids, column_name, 
                                             '1_TW_9', test_step= 'stale country dataset - channels')

# 


# In[19]:


tw_country_df.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv", 
                     index=None, na_rep='')


# In[ ]:





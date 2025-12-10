#!/usr/bin/env python
# coding: utf-8

# country isn't updated anymore so we are using an old year 

# In[1]:


platformID = 'TWI'


# In[2]:


from datetime import datetime
import pandas as pd

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


platformID


# In[5]:


# country
country_cols = ['PlaceID', 'TWI_CountryName']
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)[country_cols]

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

### RUN TESTS
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_country_0", f"{platformID}_country_1", f"{platformID}_country_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_country_3", f"{platformID}_country_4", f"{platformID}_country_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_country_6", f"{platformID}_country_7", f"{platformID}_country_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# # ingestion

# # country

# In[6]:


# takes forever to load
cols = ['Week Number', 'TW Account ID', 'Country', 'Weekly Engagements', 'Engagement %', 'Weekly Video Views']
tw_country_raw = pd.read_excel(f'../data/stale/stale_Twitter Engagements inc country.xlsx')[cols]
# ensure the accounts are strings (TW & FB)
tw_country_raw['TW Account ID'] = tw_country_raw['TW Account ID'].astype(str)

tw_country_raw = tw_country_raw.rename(columns={"TW Account ID": "Channel ID", 
                                              "Week Number": "WeekNumber_finYear",
                                               'Country': 'TWI_CountryName'})


# In[7]:


tw_country_raw['TWI_CountryName'] = tw_country_raw['TWI_CountryName'].replace("Other", "Unknown").fillna('Unknown')

tw_country_df = tw_country_raw.merge(country_codes, on='TWI_CountryName', how='left')
test_functions.test_inner_join(tw_country_raw, country_codes, 
                               ['TWI_CountryName'], 
                               f"{platformID}_country_9", 
                               test_step='checking country in lookup',
                               focus='left')

# add service 
tw_country_df = tw_country_df.merge(socialmedia_accounts[['Channel ID', 'ServiceID', 
                                                          'Excluding UK', 'Linked FB Account']], 
                                    on='Channel ID', how='left')
test_functions.test_inner_join(tw_country_df, socialmedia_accounts[['Channel ID', 'ServiceID',
                                                          'Excluding UK', 'Linked FB Account']], 
                               ['Channel ID'], 
                               f"{platformID}_country_11", 
                               test_step='checking service join in lookup',
                               focus='left')

# test accounts 
column_name = 'Channel ID'
test_functions.test_filter_elements_returned(tw_country_df, channel_ids, column_name, 
                                             f'{platformID}_country_10', 
                                             test_step= 'stale country dataset - channels')


# In[8]:


tw_country_df.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv", 
                     index=None, na_rep='')


# In[10]:


tw_country_df['Channel ID'].unique()


# In[ ]:





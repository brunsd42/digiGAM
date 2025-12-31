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

from functions import lookup_loader, execute_sql_query
import test_functions


# In[4]:


lookup = lookup_loader(gam_info, platformID, '1c',
                       with_country=True, country_col=['TWI_CountryName'])
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']
country_codes = lookup['country_codes']


# # ingestion

# # country

# In[5]:


# takes forever to load
cols = ['Week Number', 'TW Account ID', 'Country', 'Weekly Engagements', 'Engagement %', 'Weekly Video Views']
tw_country_raw = pd.read_excel(f'../data/stale/stale_Twitter Engagements inc country.xlsx')[cols]
# ensure the accounts are strings (TW & FB)
tw_country_raw['TW Account ID'] = platformID+tw_country_raw['TW Account ID'].astype(str)

tw_country_raw = tw_country_raw.rename(columns={"TW Account ID": "Channel ID", 
                                              "Week Number": "WeekNumber_finYear",
                                               'Country': 'TWI_CountryName'})


# In[6]:


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()

tw_country_raw['TWI_CountryName'] = tw_country_raw['TWI_CountryName'].replace("Other", "Unknown").fillna('Unknown')

tw_country_df = tw_country_raw.merge(country_codes, on='TWI_CountryName', how='left')
test_functions.test_inner_join(tw_country_raw, country_codes, 
                               ['TWI_CountryName'], 
                               f"{platformID}_1c_09", 
                               test_step='checking country in lookup',
                               focus='left')

# add service 
tw_country_df = tw_country_df.merge(socialmedia_accounts[['Channel ID', 'ServiceID', 
                                                          'Excluding UK', 'Linked FB Account']], 
                                    on='Channel ID', how='left')
test_functions.test_inner_join(tw_country_df, socialmedia_accounts[['Channel ID', 'ServiceID',
                                                          'Excluding UK', 'Linked FB Account']], 
                               ['Channel ID'], 
                               f"{platformID}_1c_10", 
                               test_step='checking service join in lookup',
                               focus='left')

# test accounts 
column_name = 'Channel ID'
test_functions.test_filter_elements_returned(tw_country_df, channel_ids, column_name, 
                                             f'{platformID}_1c_11', 
                                             test_step= 'stale country dataset - channels')


# In[8]:


tw_country_df.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv", 
                     index=None, na_rep='')


# In[ ]:





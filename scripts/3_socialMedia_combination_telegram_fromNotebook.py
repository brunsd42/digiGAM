#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'TEL'


# ## Status 
# twitter data is currently ingested and the process of adding facebook factors to it has been started, however the output files still show significant differences to minnie's dataset. 
# next step here is retesting the input files and then step by step through the combination process. 
# 
# twitter business unit and aggregated services is currently calculated by using minnie's dataset (helper/tw_minnie_preBU.csv)

# In[2]:


import os 
from datetime import datetime
import pandas as pd
import numpy as np
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

from functions import execute_sql_query, gnl_expander
import test_functions


# In[4]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

#socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

### RUN TESTS
test_functions.test_lookup_files(country_codes, ['PlaceID'], [f"{platformID}_3_0", f"{platformID}_3_1", f"{platformID}_3_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3_3", f"{platformID}_3_4", f"{platformID}_3_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_3_6", f"{platformID}_3_7", f"{platformID}_3_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")



# # ingestion

# In[5]:


telegram_df = pd.read_excel(f"../data/raw/{platformID}/telegram-weekly-reach-gam-2026.xlsx")
telegram_df['Channel ID'] = telegram_df['ServiceID']
cols = ['w/c', 'ServiceID', 'PlaceID', 'Channel ID', 'Reach']
telegram_df = telegram_df[cols].rename(columns={'Reach': 'uv_by_country'})
telegram_df = telegram_df[telegram_df['ServiceID'].isin(socialmedia_accounts['ServiceID'].unique())]
# rename stuff
telegram_df.head()


# In[6]:


telegram_df['ServiceID'].unique()


# In[7]:


# testing 
# missing page_ids
test_functions.test_filter_elements_returned(telegram_df, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_3_9",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='Channel ID',
                                               main_data=telegram_df,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"{platformID}_3_10",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(telegram_df, 
                           numeric_columns=['uv_by_country'], 
                           test_number=f"{platformID}_3_11",
                           test_step='Check no missing values in page fans column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(telegram_df, ['Channel ID', 'w/c', 'PlaceID'], 
                               test_number=f"{platformID}_3_12",
                               test_step='Check no duplicates from redshift returned')

# test for countries being found
test_functions.test_inner_join(telegram_df, 
                               country_codes, 
                               ['PlaceID'], 
                               f"{platformID}_3_13",
                               test_step='calculating country %',
                               focus='left')


# In[9]:


path = f"../data/processed/{platformID}"
os.makedirs(path, exist_ok=True)
print(telegram_df.shape)
cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
telegram_df[cols].to_csv(f"{path}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv",
                        index=None)


# In[ ]:





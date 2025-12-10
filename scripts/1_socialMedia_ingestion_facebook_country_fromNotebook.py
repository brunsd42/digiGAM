#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'FBE'


# In[2]:


from datetime import datetime
import pandas as pd

import os 


# ## import helper 

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
from functions import execute_sql_query, compare_or_update_reference
import test_functions

from config import gam_info


# In[4]:


# country
country_cols = ['YT-_FBE_codes', 'PlaceID']
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID',
                             keep_default_na=False)[country_cols]
# week 
week_cols = ['w/c']
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accoutns
channel_cols=['Channel ID']
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()

### RUN TESTS
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_1country_0", f"{platformID}_1country_1", f"{platformID}_1country_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_1country_3", f"{platformID}_1country_4", f"{platformID}_1country_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_1country_6", f"{platformID}_1country_7", f"{platformID}_1country_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# ## country

# In[5]:


sql_query = f"""
    SELECT 
        week_commencing,
        page_id ,
        page_name,
        page_fans_country_total AS page_fans_country,
        country_code AS fb_metric_breakdown
    FROM
        redshiftdb.central_insights.adverity_social_facebook_page_fans_by_country
    WHERE
        week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_redshift_extract.csv"

df = execute_sql_query(sql_query)
df['page_id'] = df['page_id'].astype(str)
df.to_csv(file, index=False, na_rep='')

facebook_country_raw = pd.read_csv(file, keep_default_na=False)

facebook_country_raw['page_id'] = facebook_country_raw['page_id'].astype(str)
facebook_country_raw['week_commencing'] = pd.to_datetime(facebook_country_raw['week_commencing'])
facebook_country_raw = facebook_country_raw.rename(columns={'page_id': 'Channel ID',
                                                            'page_name': 'Channel Name',
                                                            'week_commencing': 'w/c',
                                                            'fb_metric_breakdown': 'YT-_FBE_codes'})

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(facebook_country_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"9_{platformID}_country",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='Channel ID',
                                               main_data=facebook_country_raw,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"10_{platformID}_country",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(facebook_country_raw, 
                           numeric_columns=['page_fans_country'], 
                           test_number=f"11_{platformID}_country",
                           test_step='Check no missing values in page fans column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(facebook_country_raw, ['Channel ID', 'w/c'], 
                               test_number=f"12_{platformID}_country",
                               test_step='Check no duplicates from redshift returned')


# In[6]:


# filter to relevant pageID's
facebook_country = facebook_country_raw[facebook_country_raw['Channel ID'].isin(channel_ids)]

test_functions.test_inner_join(facebook_country, 
                               country_codes, 
                               ['YT-_FBE_codes'], 
                               f"13_{platformID}_country",
                               test_step='calculating country %')

facebook_country = facebook_country.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], 
                                          on='YT-_FBE_codes', 
                                          how='left').drop(columns=['YT-_FBE_codes'])

# Group by specified columns and sum the fb_metric_value
facebook_country_sum = facebook_country.groupby(['Channel ID', 'w/c'])\
                                        .agg(Sum_page_fans_country=('page_fans_country', 'sum'))\
                                        .reset_index()
facebook_country = facebook_country.merge(facebook_country_sum, 
                                          how='inner', on= ['Channel ID', 'w/c'])
test_functions.test_inner_join(facebook_country, facebook_country_sum, 
                               ['Channel ID', 'w/c'], 
                               f"14_{platformID}_country", 
                               test_step='calculating country %')

facebook_country['country_%'] = facebook_country['page_fans_country']/facebook_country['Sum_page_fans_country']

### RUN TESTS
test_functions.test_percentage(facebook_country, 
                               ['Channel ID', 'w/c'], 
                               f"15_{platformID}_country", 
                               test_step='summing country % per week & account')


# In[7]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
facebook_country[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv",
                        index=None)
'''
compare_or_update_reference(facebook_country[cols], 
                            "../test/refactoring/fbe_country_expected.pkl", 
                            cols, update=True)
'''


# In[ ]:





# In[ ]:





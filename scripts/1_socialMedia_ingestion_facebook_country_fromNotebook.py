#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd


# ## import helper 

# In[2]:


platformID = 'FBE'


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


# In[9]:


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

socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
#socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# ## country

# In[16]:


sql_query = f"""
    SELECT 
        page_id AS fb_page_id,
        page_name AS fb_page_name,
        page_fans_country_total AS page_fans_country,
        country_code AS fb_metric_breakdown,
        week_commencing
    FROM
        redshiftdb.central_insights.adverity_social_facebook_page_fans_by_country
    WHERE
        week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_redshift_extract.csv"


df = execute_sql_query(sql_query)
df['fb_page_id'] = df['fb_page_id'].astype(str)
df.to_csv(file, index=False, na_rep='')

facebook_country_raw = pd.read_csv(file, keep_default_na=False)

facebook_country_raw['fb_page_id'] = facebook_country_raw['fb_page_id'].astype(str)
facebook_country_raw['week_commencing'] = pd.to_datetime(facebook_country_raw['week_commencing'])
facebook_country_raw = facebook_country_raw.rename(columns={'week_commencing': 'w/c',
                                                            'fb_metric_breakdown': 'YT-_FBE_codes'})

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(facebook_country_raw, 
                                             channel_ids, 
                                             'fb_page_id', 
                                             f"1_{platformID}_country",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='fb_page_id',
                                               main_data=facebook_country_raw,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"2_{platformID}_country",
                                               test_step="Check all weeks present for each account")




# In[22]:


# filter to relevant pageID's
facebook_country = facebook_country_raw[facebook_country_raw['fb_page_id'].isin(channel_ids)]

test_functions.test_inner_join(facebook_country, 
                               country_codes[['YT-_FBE_codes', 'PlaceID']], 
                               ['YT-_FBE_codes'], 
                               f"3_{platformID}_country",
                               test_step='calculating country %')

facebook_country = facebook_country.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], 
                                          on='YT-_FBE_codes', 
                                          how='left').drop(columns=['YT-_FBE_codes'])

# Group by specified columns and sum the fb_metric_value
facebook_country_sum = facebook_country.groupby(['fb_page_id', 'w/c'])\
                                        .agg(Sum_page_fans_country=('page_fans_country', 'sum'))\
                                        .reset_index()
facebook_country = facebook_country.merge(facebook_country_sum, 
                                          how='inner', on= ['fb_page_id', 'w/c'])
test_functions.test_inner_join(facebook_country, facebook_country_sum, 
                               ['fb_page_id', 'w/c'], 
                               f"4_{platformID}_country", 
                               test_step='calculating country %')

facebook_country['country_%'] = facebook_country['page_fans_country']/facebook_country['Sum_page_fans_country']

### RUN TESTS
test_functions.test_percentage(facebook_country, 
                               ['fb_page_id', 'w/c'], 
                               f"5_{platformID}_country", 
                               test_step='summing country % per week & account')


# In[28]:


cols = ['fb_page_id', 'fb_page_name', 'w/c', 'PlaceID', 'country_%']
facebook_country[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv",
                        index=None)
'''
compare_or_update_reference(facebook_country[cols], 
                            "../test/refactoring/fbe_country_expected.pkl", 
                            cols, update=True)
'''


# In[ ]:





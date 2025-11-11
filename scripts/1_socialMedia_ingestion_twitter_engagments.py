#!/usr/bin/env python
# coding: utf-8

# In[4]:


platformID = 'TWI'


# In[5]:


from datetime import datetime
import pandas as pd

import psycopg2


# ## import helper

# In[6]:


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


# In[7]:


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
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# # ingestion

# ## activity

# In[8]:


metric_ids = ['tweet media engagements', 
                 'tweet engagements',
                 'video_minutes_viewed',
                 'video_video_views',
                 'video_playback_25',
                 'video_playback_50',
                ]
formatted_metric_ids = ', '.join(f"'{metric_id}'" for metric_id in metric_ids)

sql_query = f"""
    SELECT 
        tw_account_id, 
        tw_metric_id, 
        tw_metric_period, 
        tw_metric_end_time, 
        tw_metric_breakdown, 
        tw_metric_value 
    FROM
        redshiftdb.central_insights.tw_account_insights 
    WHERE
        (tw_metric_id IN ({formatted_metric_ids})
        AND
        tw_account_id IN ({formatted_channel_ids}) 
        AND 
        tw_metric_period = 'weekdiff' 
        AND 
        tw_metric_end_time between '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}') ;
"""
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_activity_redshift_extract.csv"
df = execute_sql_query(sql_query)
df.to_csv(file, index=False, na_rep='')

twitter_activity_raw = pd.read_csv(file, keep_default_na=False)
twitter_activity_raw['tw_account_id'] = twitter_activity_raw['tw_account_id'].apply(lambda x: str(int(x)))

column_name = 'tw_metric_id'
test_functions.test_filter_elements_returned(twitter_activity_raw, metric_ids, column_name, 
                                             '1_TW_1', test_step= 'sql query tw_accounts_insights - metrics')
column_name = 'tw_account_id'
test_functions.test_filter_elements_returned(twitter_activity_raw, channel_ids, column_name, 
                                             '1_TW_2', test_step= 'sql query tw_accounts_insights - channels')

twitter_activity_raw['tw_metric_end_time'] = pd.to_datetime(twitter_activity_raw['tw_metric_end_time'])
test_functions.test_weeks_presence('week_ending', 
                                    twitter_activity_raw.rename(columns={'tw_metric_end_time': 'week_ending'}), 
                                    week_tester, '1_TW_3', "tw_account_insights sql query")



# In[9]:


# Perform crosstab operation
twitter_activity = pd.pivot_table(twitter_activity_raw, 
                             values='tw_metric_value', 
                             index=['tw_account_id', 'tw_metric_period', 'tw_metric_end_time', 'tw_metric_breakdown'], 
                             columns='tw_metric_id', 
                             aggfunc='sum').reset_index()

# test there is now one row per channel / week
test_functions.test_duplicates(twitter_activity, ['tw_account_id', 'tw_metric_end_time'], 
                               "1_TW_4", "reshaping twitter data to get metric per week & channel")

test_functions.test_weeks_presence_per_account('week_ending', 'tw_account_id', 
                                               twitter_activity.rename(columns={'tw_metric_end_time': 'week_ending'}), 
                                               week_tester, '1_TW_5', test_step='reshaping activity data')


# ## metadata

# In[10]:


sql_query = f"""
    SELECT 
        tw_account_id, 
        tw_account_name, 
        tw_account_username, 
        tw_account_bbc_clean_name, 
        tw_account_bbc_bus_unit, 
        week_ending, 
        week_commencing
    FROM
        redshiftdb.central_insights.tw_account_metadata
    WHERE
        tw_account_id IN ({formatted_channel_ids}) 
        AND
        week_ending Between '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}'
;
"""
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_metadata_redshift_extract.csv"
df = execute_sql_query(sql_query)
df.to_csv(file, index=False, na_rep='')

twitter_metadata = pd.read_csv(file, keep_default_na=False)
twitter_metadata['tw_account_id'] = twitter_metadata['tw_account_id'].apply(lambda x: str(int(x)))
twitter_metadata['week_ending'] = pd.to_datetime(twitter_metadata['week_ending'])
column_name = 'tw_account_id'
test_functions.test_filter_elements_returned(twitter_metadata, channel_ids, column_name, 
                                             '1_TW_6', test_step= 'sql query tw_account_metadata - channels')

test_functions.test_weeks_presence('week_ending', 
                                    twitter_activity_raw.rename(columns={'tw_metric_end_time': 'week_ending'}), 
                                    week_tester, '1_TW_7', "tw_account_metadata sql query")


# ## combine

# In[11]:


twitter_activity = twitter_activity.rename(columns={'tw_metric_end_time': 'week_ending', 
                                                    'week_commencing': 'w/c'}, 
                                                    )
twitter_activity_metadata = twitter_activity.merge(twitter_metadata, how='inner',
                                    on=['tw_account_id', 'week_ending'])

test_functions.test_inner_join(twitter_activity, twitter_metadata, ['tw_account_id', 'week_ending'],
                               '1_TW_8', test_step="combining activity & metadata ")


# In[12]:


# adress those are lost 
twitter_activity_metadata.drop_duplicates().to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv")

twitter_activity_metadata['week_ending'] = pd.to_datetime(twitter_activity_metadata['week_ending'])
twitter_activity_metadata = twitter_activity_metadata.merge(week_tester[['week_ending', 'WeekNumber_finYear']], on='week_ending', how='left')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

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


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')
#socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# # ingestion

# ## activity

# In[5]:


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

twitter_activity_raw = twitter_activity_raw.rename(columns={'tw_metric_end_time': 'week_ending'})
twitter_activity_raw = twitter_activity_raw.merge(week_tester[['w/c', 'WeekNumber_finYear', 'week_ending']], on='week_ending', how='left', 
                       indicator=True)

print(twitter_activity_raw._merge.value_counts())
twitter_activity_raw = twitter_activity_raw.drop(columns=['_merge', 'week_ending'])


# In[6]:


# Perform crosstab operation
twitter_activity = pd.pivot_table(twitter_activity_raw, 
                             values='tw_metric_value', 
                             index=['tw_account_id', 'tw_metric_period', 
                                    'w/c', 'WeekNumber_finYear',
                                    'tw_metric_breakdown'], 
                             columns='tw_metric_id', 
                             aggfunc='sum').reset_index()

# test there is now one row per channel / week
test_functions.test_duplicates(twitter_activity, ['tw_account_id', 'w/c'], 
                               "1_TW_4", "reshaping twitter data to get metric per week & channel")

test_functions.test_weeks_presence_per_account('w/c', 'tw_account_id', twitter_activity, 
                                               week_tester, '1_TW_5', 
                                               test_step='reshaping activity data')



# In[7]:


cols = ['tw_account_id', 'w/c', 'WeekNumber_finYear', 'tw_metric_breakdown',
       'tweet engagements', 'tweet media engagements', 'video_minutes_viewed',
       'video_playback_25', 'video_playback_50', 'video_video_views']
# adress those are lost 
twitter_activity[cols].drop_duplicates().to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv", 
                                                index=None)


# In[8]:


twitter_activity.head()


# In[ ]:





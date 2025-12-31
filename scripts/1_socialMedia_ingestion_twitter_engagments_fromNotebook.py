#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'TWI'


# In[2]:


from datetime import datetime
import pandas as pd

import psycopg2

import os


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

from functions import lookup_loader, execute_sql_query, compare_or_update_reference
import test_functions


# In[4]:


lookup = lookup_loader(gam_info, platformID, '1e')

week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']


# # ingestion

# ## activity

# In[5]:


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

sql_query = f"""
    SELECT 
        week_commencing,
        account_id,
        SUM(engagements) AS tweet_engagements,
        SUM(video_views) AS video_video_views
    FROM
        redshiftdb.central_insights.adverity_social_x_by_tweets
    WHERE
        week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
    GROUP BY
        week_commencing,
        account_id
    ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_activity_redshift_extract.csv"

try: 
    df = execute_sql_query(sql_query)
    df['account_id'] = platformID+ df['account_id'].astype(str)
    df.to_csv(file, index=False, na_rep='')
except:
    print("no redshift connection using last time queried")

twitter_activity_raw = pd.read_csv(file, keep_default_na=False)
twitter_activity_raw['account_id'] = twitter_activity_raw['account_id']
twitter_activity_raw['week_commencing'] = pd.to_datetime(twitter_activity_raw['week_commencing'])
twitter_activity = twitter_activity_raw.rename(columns={'account_id': 'Channel ID',
                                                            'week_commencing': 'w/c'})

# keep only measured accounts
print(twitter_activity.shape)
twitter_activity = twitter_activity.merge(socialmedia_accounts[['Channel ID', 'ServiceID']], 
                                          on='Channel ID', how='right')
print(twitter_activity.shape)

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(twitter_activity, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_1e_06",
                                             "Check that all page IDs are found in SQL")

# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col='Channel ID',
                                               main_data=twitter_activity,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts,
                                               test_number=f"{platformID}_1e_07",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(twitter_activity, 
                           numeric_columns=['tweet_engagements', 'video_video_views'], 
                           test_number=f"{platformID}_1e_08",
                           test_step='Check no missing values in metric columns column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(twitter_activity, ['Channel ID', 'w/c'], 
                               test_number=f"{platformID}_1e_09",
                               test_step='Check no duplicates from redshift returned')


# In[6]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

twitter_activity.to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                        index=None)
'''
compare_or_update_reference(twitter_activity[cols], 
                            f"../test/refactoring/{platformID}_expected.pkl", 
                            cols, update=False)
'''


# In[ ]:





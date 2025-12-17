#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
import numpy as np
import os 


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
from functions import execute_sql_query
import test_functions

from config import gam_info


# In[4]:


# week 
week_cols = ['w/c']
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')

# social media accounts
channel_cols=['Channel ID']
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

# socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)

### RUN TESTS
test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_1engage_3", f"{platformID}_1engage_4", f"{platformID}_1engage_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_1engage_6", f"{platformID}_1engage_7", f"{platformID}_1engage_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# # engagements 

# In[5]:


sql_query = f"""
    SELECT
        week_commencing,
        page_id,
        CASE
            WHEN (AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user) 
                 > AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer)
            THEN ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user)) 
                 + (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer))*0.04827
            ELSE (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer)) 
                 + ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user))*0.04822
        END AS engaged_reach
    FROM 
        redshiftdb.central_insights.adverity_social_facebook_by_page AS p
    RIGHT JOIN
        world_service_audiences_insights.social_media_lookup_fb AS l
        ON p.page_id = l.fb_page_id
    WHERE 
        period = 'week'
    AND
        week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
    GROUP BY 
        week_commencing, page_id
        ;
"""

file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_engagements_redshift_extract.csv"

#df = execute_sql_query(sql_query)
#df['page_id'] = df['page_id'].astype(str)
#df.to_csv(file, index=False, na_rep='')

facebook_engagements_raw = pd.read_csv(file, keep_default_na=False)
facebook_engagements_raw['page_id'] = facebook_engagements_raw['page_id'].astype(str)
facebook_engagements_raw['week_commencing'] = pd.to_datetime(facebook_engagements_raw['week_commencing'])
facebook_engagements_raw = facebook_engagements_raw.rename(columns={'page_id': 'Channel ID', 
                                                                    'week_commencing': 'w/c'})
print(facebook_engagements_raw.shape)

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(facebook_engagements_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"6_{platformID}_engagements",
                                             "Check that all page IDs are found in SQL")

# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='Channel ID',
                                               main_data=facebook_engagements_raw,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"7_{platformID}_engagements",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(facebook_engagements_raw, 
                           numeric_columns=['engaged_reach'], 
                           test_number=f"8_{platformID}_engagements",
                           test_step='Check no missing values in engaged_reach column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(facebook_engagements_raw, 
                               ['Channel ID', 'w/c'], 
                               test_number=f"9_{platformID}_engagements",
                               test_step='Check no duplicates from redshift returned')


# ## processing engagements

# In[6]:


facebook_engagements = facebook_engagements_raw.merge(socialmedia_accounts[['Channel ID', 'ServiceID']], 
                                                      on='Channel ID', how='left', indicator=True)
test_functions.test_inner_join(facebook_engagements_raw, socialmedia_accounts, 
                               ['Channel ID'], 
                               f"10_{platformID}_engagements", 
                               test_step='checking social media accounts in lookup, adding service',
                               focus='left')


# In[7]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'ServiceID', 'w/c', 'engaged_reach']
facebook_engagements[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                                       index=None)


# In[8]:


facebook_engagements[facebook_engagements['ServiceID'].isin(['BNI', 'BNO', 'GNL'])]


# In[10]:


facebook_engagements[(facebook_engagements['w/c'] == '2025-12-01')  & 
    (facebook_engagements['Channel ID'].isin(['630866223444617']))]


# In[ ]:





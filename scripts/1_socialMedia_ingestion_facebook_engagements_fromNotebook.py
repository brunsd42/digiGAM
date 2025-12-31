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
from functions import lookup_loader, execute_sql_query
import test_functions

from config import gam_info


# In[4]:


lookup = lookup_loader(gam_info, platformID, '1e',
                       with_country=False)
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']


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

# TODO make this more fail safe - e.g. I want to run this and say if it should query redshift or not
try: 
    df = execute_sql_query(sql_query)
    df['page_id'] = platformID+df['page_id'].astype(str)
    df.to_csv(file, index=False, na_rep='')
except:
    print("no redshift connection using last time queried")
    
facebook_engagements_raw = pd.read_csv(file, keep_default_na=False)
facebook_engagements_raw['page_id'] = facebook_engagements_raw['page_id'].astype(str)
facebook_engagements_raw['week_commencing'] = pd.to_datetime(facebook_engagements_raw['week_commencing'])
facebook_engagements_raw = facebook_engagements_raw.rename(columns={'page_id': 'Channel ID', 
                                                                    'week_commencing': 'w/c'})
print(facebook_engagements_raw.shape)


# In[7]:


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(facebook_engagements_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_1e_06",
                                             "Check that all page IDs are found in SQL")

# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col='Channel ID',
                                               main_data=facebook_engagements_raw,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts[['Channel ID', 'Start', 'End']],
                                               test_number=f"{platformID}_1e_07",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(facebook_engagements_raw, 
                           numeric_columns=['engaged_reach'], 
                           test_number=f"{platformID}_1e_08",
                           test_step='Check no missing values in engaged_reach column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(facebook_engagements_raw, 
                               ['Channel ID', 'w/c'], 
                               test_number=f"{platformID}_1e_09",
                               test_step='Check no duplicates from redshift returned')


# ## processing engagements

# In[8]:


# needs to be right because we want to make sure that all the accounts in lookup are there 
facebook_engagements = facebook_engagements_raw.merge(socialmedia_accounts[['Channel ID', 'ServiceID', 'Start', 'End']], 
                                                      on='Channel ID', how='right', indicator=True)
test_functions.test_inner_join(facebook_engagements_raw, socialmedia_accounts, 
                               ['Channel ID'], 
                               f"{platformID}_1e_10", 
                               test_step='checking social media accounts in lookup, adding service',
                               focus='right')


mask = (
    (facebook_engagements['w/c'] >= facebook_engagements['Start']) &
    (
        facebook_engagements['End'].isna() |
        (facebook_engagements['w/c'] <= facebook_engagements['End'])
    )
)
# there are newly active accounts that have 0 for the weeks before they became active
# for accurate reach calculation those weeks need to be removed
facebook_engagements = facebook_engagements[mask]


# In[9]:


facebook_engagements[facebook_engagements['Channel ID'] == 'FBE102775241582303'].sort_values('w/c')


# In[10]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'ServiceID', 'w/c', 'engaged_reach']
facebook_engagements[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                                       index=None)


# In[ ]:





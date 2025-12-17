#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'INS'


# In[2]:


from datetime import datetime
import pandas as pd
import numpy as np
import os 

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


# In[10]:


# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'IG FB Linked ID': 'str'}
socialmedia_accounts = pd.read_excel("../helper/ins_account_lookup.xlsx", dtype=dtype_dict)
channel_ids = socialmedia_accounts['Channel ID'].tolist()

# Factors
ins_factors = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='INS_Factors', index_col='ServiceID')['Factor']

### RUN TESTS
test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_engage_0", f"{platformID}_engage_1", f"{platformID}_engage_2"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_engage_3", f"{platformID}_engage_4", f"{platformID}_engage_5"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# # ingestion

# # activity

# In[5]:


sql_query = f"""
    SELECT
        p.week_commencing,
        l.ig_account_id as ig_user_id,
        GREATEST(0, SUM(COALESCE(p.engagements_week_diff, p.engagements))) AS engagements,
        GREATEST(0, SUM(COALESCE(p.media_views_week_diff, p.video_views))) AS impressions,
        GREATEST(0, SUM(
            CASE
                WHEN UPPER(p.media_type) = 'VIDEO' THEN COALESCE(p.media_views_week_diff, p.video_views)
                ELSE 0
            END
        )) AS media_views,
        GREATEST(0, MAX(r.weekly_reach)) AS weekly_reach
    FROM
        central_insights.adverity_social_instagram_by_posts AS p
    RIGHT JOIN
            world_service_audiences_insights.social_media_lookup_ig AS l
        ON 
            p.account_id = l.ig_account_id
    LEFT JOIN
            central_insights.adverity_social_instagram_by_reach AS r
        ON 
            p.week_commencing = r.week_commencing
        AND 
            p.account_id = r.account_id
    WHERE
            p.week_commencing IS NOT NULL
        AND
            p.week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
    GROUP BY
        p.week_commencing,
        l.ig_account_id
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_activity_redshift_extract.csv"

#df = execute_sql_query(sql_query)
#df['ig_user_id'] = df['ig_user_id'].astype(str) 
#df.to_csv(file, index=False, na_rep='')

ig_views_raw = pd.read_csv(file, keep_default_na=False)
ig_views_raw['ig_user_id'] = ig_views_raw['ig_user_id'].astype(str) 
ig_views_raw['week_commencing'] = pd.to_datetime(ig_views_raw['week_commencing'])
ig_views_raw = ig_views_raw.rename(columns={'ig_user_id': 'Channel ID',
                                            'week_commencing': 'w/c'})

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(ig_views_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_engage_6", 
                                             "Check that all page IDs are found in SQL")
# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               id_column='Channel ID',
                                               main_data=ig_views_raw,
                                               week_lookup=week_tester[['w/c']],
                                               test_number=f"{platformID}_engage_7", 
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(ig_views_raw, 
                           numeric_columns=['engagements', 'impressions', 'media_views', 'weekly_reach'], 
                           test_number=f"{platformID}_engage_8", 
                           test_step='Check no missing values in metric columns from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(ig_views_raw, 
                               ['Channel ID', 'w/c'], 
                               test_number=f"{platformID}_engage_9", 
                               test_step='Check no duplicates from redshift returned')

# General outlier check
test_functions.test_outliers_general(ig_views_raw,
                      numeric_columns=['engagements', 'impressions', 'media_views', 'weekly_reach'],
                      test_number=f"{platformID}_engage_10", 
                      test_step='Check for extreme outliers in metrics')

# Outlier vs reference
reference_df = pd.read_excel("../data/stale/IG Profile Engagement Metrics - Weekly GAM 2026.xlsx")\
    .rename(columns={'IG Page ID': 'Channel ID',
                     'Engagement': 'engagements',
                     'Reach': 'weekly_reach',
                     'Impressions': 'impressions',
                      'views': 'media_views',
                    })
reference_df['Channel ID'] = reference_df['Channel ID'].dropna().apply(lambda x: str(int(x)))
reference_df['w/c'] = pd.to_datetime(reference_df['w/c'])
test_functions.test_outliers_vs_reference(ig_views_raw, reference_df,
                             key_columns=['Channel ID', 'w/c'],
                             numeric_columns=['engagements', 'impressions', 'media_views', 'weekly_reach'],
                             test_number=f"{platformID}_engage_11",
                             test_step='Compare metrics to reference period')



# ## stale - temporary fix

# In[6]:


path = "../data/stale/IG Profile Engagement Metrics - Weekly GAM 2026.xlsx"
stale_engagements = pd.read_excel(path)[['w/c', 'IG Page ID', 'Engagement', 'Reach', 'Impressions', 'views']]
stale_engagements = stale_engagements.rename(columns={
    "IG Page ID": "Channel ID",
    "Engagement": "engagements",
    "Reach": "weekly_reach",
    "Impressions": "impressions",
    "views": "media_views"
})
stale_engagements['Channel ID'] = stale_engagements['Channel ID'].dropna().apply(lambda x: str(int(x)))
stale_engagements['w/c'] = pd.to_datetime(stale_engagements['w/c'])

until_date = "2025-11-10"
stale_engagements = stale_engagements[stale_engagements['w/c'] < until_date]

ig_views_raw = pd.concat([stale_engagements, ig_views_raw[ig_views_raw['w/c'] >= until_date]],
                        ignore_index=True)


# In[7]:


# add data different to reference sheet

# missing week? replace with value from reference


# In[8]:


ig_views_slim = ig_views_raw.merge(socialmedia_accounts[['Channel ID', 'ServiceID']], 
                                                      on='Channel ID', how='left')
test_functions.test_inner_join(ig_views_raw, socialmedia_accounts, 
                               ['Channel ID'], 
                               f"{platformID}_engage_12",  
                               test_step='checking social media accounts in lookup, adding service',
                               focus='left')


plays_factor = pd.read_excel("../data/stale/Instagram - Views to Reels Plays.xlsx", 
                             dtype={'IG Page ID': 'str'})\
                    .rename(columns={'IG Page ID': 'Channel ID'})[['Channel ID', 'reels_replay_factor']]


# In[13]:


test_functions.test_inner_join(ig_views_slim, plays_factor,
                               ['Channel ID'],
                               f"{platformID}_engage_13", 
                               test_step='adding views to reels plays', focus='left')
ig_views = ig_views_slim.merge(plays_factor, on=['Channel ID'], how='left')
ig_views['reels_replay_factor'] = ig_views['reels_replay_factor'].fillna(plays_factor['reels_replay_factor'].mean())

# missing values per week / page id 
test_functions.test_non_null_and_positive(ig_views, 
                           numeric_columns=['reels_replay_factor'], 
                           test_number=f"{platformID}_engage_14", 
                           test_step='Check no missing values in reels_replay_factor')
### 
metrics = ['engagements', 'media_views', 'impressions', 'weekly_reach']
for metric in metrics:
    ig_views[metric] = ig_views[metric].fillna(0)

ig_views['plays'] = ig_views.apply(lambda r: r['media_views'] / r['reels_replay_factor'] 
                                   if r['reels_replay_factor'] else 0, axis=1)

# Apply the logic to create a new column 'adjusted_reels_plays'
ig_views['30 view'] = ig_views['plays'] * ig_views['ServiceID'].map(ins_factors)\
                .fillna(ins_factors['ALL'])

def safe_ratio(row):
    if row['weekly_reach'] == 0 or pd.isna(row['weekly_reach']):
        return np.nan  
    if row['ServiceID'] == 'PER':
        return (row['impressions'] / row['weekly_reach']) * 0.052990833 + 1.693711126
    else:
        return (row['impressions'] / row['weekly_reach']) * 0.21907280062318 + 1.20241835848198

ig_views['IG Modelled Factor'] = ig_views.apply(safe_ratio, axis=1)

# Compute engagement estimate
ig_views['engaged_reach'] = ((ig_views['engagements'] + ig_views['30 view']) / ig_views['IG Modelled Factor'])


# Persian special case (ignore cap)
mask_persian = (ig_views['ServiceID'] == 'PER') & (ig_views['Channel ID'] == '17841400230391592')
ig_views.loc[mask_persian, 'engaged_reach'] = ((ig_views.loc[mask_persian, 'engagements'] +
                                                ig_views.loc[mask_persian, '30 view']) /
                                                ig_views.loc[mask_persian, 'IG Modelled Factor'])

# Cap at weekly reach
ig_views['engaged_reach'] = ig_views['engaged_reach'].clip(upper=ig_views['weekly_reach'])


# In[14]:


ig_views[(ig_views['ServiceID'] == 'PER') & 
    (ig_views['w/c'] == '2025-05-05') 
    #& (engagements['PlaceID'] == 'IRN')
    ]#['uv_by_country'].sum()


# In[15]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'ServiceID', 'w/c', 'engaged_reach']
ig_views.to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_engagements_final.csv", 
                          index=None)


# In[ ]:





# In[ ]:





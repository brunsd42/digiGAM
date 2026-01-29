#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'INS'


# In[2]:


from datetime import datetime, date
import pandas as pd
import numpy as np
import os 

import psycopg2
import missingno as msno


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

from functions import lookup_loader, execute_sql_query, compare_or_update_reference, fix_country_one_percent_prev_week
import test_functions 


# In[4]:


lookup = lookup_loader(gam_info, platformID, '1c',
                       with_country=True, country_col=['YT-_FBE_codes', 'ins_country_name',])
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']
country_codes = lookup['country_codes']


# In[5]:


country_codes[country_codes['ins_country_name'] == 'Iran']


# # ingestion

# In[6]:


# keep country code where clause - the other rows are for age & gender in diff cols
sql_query = f"""
    SELECT 
        week_commencing,
        l.ig_account_id as page_id,
        page_name, 
        country_code,
        country_name as ins_country_name,
        followers_by_demographic
    FROM
        redshiftdb.central_insights.adverity_social_instagram_by_page_demo AS p
    RIGHT JOIN
            world_service_audiences_insights.social_media_lookup_ig AS l
        ON 
            p.page_id = l.ig_identifier
    WHERE
            week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
        AND
            followers_by_demographic > 0
        AND 
            country_name <> ''
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_redshift_extract.csv"

try: 
    df = execute_sql_query(sql_query)
    df['page_id'] = platformID+df['page_id'].astype(str)
    df.to_csv(file, index=False, na_rep='')
except:
    print("no redshift connection using last time queried")

ig_userCountry_raw = pd.read_csv(file, keep_default_na=False)

ig_userCountry_raw['page_id'] = ig_userCountry_raw['page_id'].astype(str) 
ig_userCountry_raw['week_commencing'] = pd.to_datetime(ig_userCountry_raw['week_commencing'])
ig_userCountry_raw = ig_userCountry_raw.rename(columns={'page_id': 'Channel ID',
                                                'page_name': 'Channel Name',
                                                'week_commencing': 'w/c',
                                                'country_code': 'YT-_FBE_codes',
                                                })


ig_userCountry_raw = ig_userCountry_raw.merge(country_codes[['YT-_FBE_codes', 'ins_country_name']], 
                                          on='ins_country_name', suffixes=['', '_name_based'],
                                          how='left').drop(columns=['ins_country_name'])
ig_userCountry_raw.loc[ig_userCountry_raw['YT-_FBE_codes'] == '', 'YT-_FBE_codes'] = ig_userCountry_raw.loc[ig_userCountry_raw['YT-_FBE_codes'] == '', 
    'YT-_FBE_codes_name_based']

ig_userCountry_raw = ig_userCountry_raw.drop(columns=['YT-_FBE_codes_name_based']).drop_duplicates()

### RUN TESTS
channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
# missing page_ids
test_functions.test_filter_elements_returned(ig_userCountry_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_1c_09",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col=['Channel ID'],
                                               main_data=ig_userCountry_raw,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts[['Channel ID', 'Start', 'End']],
                                               test_number=f"{platformID}_1c_10",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(ig_userCountry_raw, 
                           numeric_columns=['followers_by_demographic'], 
                           test_number=f"{platformID}_1c_11",
                           test_step='Check no missing values in followers_by_demographic column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(ig_userCountry_raw, ['Channel ID', 'w/c', 'YT-_FBE_codes'], 
                               test_number=f"{platformID}_1c_12",
                               test_step='Check no duplicates from redshift returned')



# In[7]:


test_functions.test_inner_join(ig_userCountry_raw, socialmedia_accounts, 
                               ['Channel ID'], 
                               f"{platformID}_1c_13",
                               test_step='checking social media accounts in lookup, adding service',
                               focus='left')
ig_userCountry = ig_userCountry_raw.merge(socialmedia_accounts[['Channel ID', 'ServiceID']], 
                                                      on='Channel ID', how='left')


# In[8]:


test_functions.test_inner_join(ig_userCountry, 
                               country_codes, 
                               ['YT-_FBE_codes'], 
                               f"{platformID}_1c_14",
                               test_step='calculating country %', 
                                focus='left')
ig_userCountry = ig_userCountry.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], 
                                          on='YT-_FBE_codes', 
                                          how='left').drop(columns=['YT-_FBE_codes'])


# In[9]:


# currently sql table has duplicates
ig_userCountry = ig_userCountry.drop_duplicates()
# test for duplicate entries 
test_functions.test_duplicates(ig_userCountry, ['Channel ID', 'w/c', 'PlaceID'], 
                               test_number=f"{platformID}_1c_15",
                               test_step='dropped duplicates past processing')


# In[10]:


grouped_df = ig_userCountry.groupby(['Channel ID', 'Channel Name', 
                                     'w/c']).agg({'followers_by_demographic': 'sum'}).reset_index()

# Rename the aggregated column
grouped_df = grouped_df.rename(columns={'followers_by_demographic': 'Sum_followers_by_demographic'})

right_cols = ['Channel ID', 'w/c', 'Sum_followers_by_demographic']
ig_country_df = ig_userCountry.merge(grouped_df[right_cols],
                                         how='inner',
                                         on=['Channel ID', 'w/c'])

test_functions.test_inner_join(ig_userCountry, grouped_df[right_cols], 
                               ['Channel ID', 'w/c'],
                               f"{platformID}_1c_16",)

ig_country_df['country_%'] = ig_country_df.apply(
    lambda row: row['followers_by_demographic'] / row['Sum_followers_by_demographic']
    if row['Sum_followers_by_demographic'] > 0 else 0, axis=1
)

test_functions.test_percentage(ig_country_df, 
                               ['Channel ID', 'w/c'], 
                               f"{platformID}_1c_17",
                               test_step='summing country % per week & account')


# In[11]:


ig_country_df.shape


# In[12]:


'''
### DANGER! FIXING IRAN REACH AT 1% DUE TO CURRENT SHUTDOWN 

def fix_country_one_percent(df, country_code, week_list):
    print(f"Fixing Iran reach at 1% due to shutdown for weeks: {week_list}")
    
    df = df.copy()

    # filter affected weeks ONLY
    affected = df[df['w/c'].isin(set(week_list))]
    print("Affected rows:", affected.shape)

    # loop only over affected weeks
    for (channel, wc), group in affected.groupby(['Channel ID', 'w/c']):
        
        # Iran row
        old_country_pct = group.loc[group['PlaceID'] == country_code, 'country_%']
        if old_country_pct.empty:
            continue

        old_country_pct = float(old_country_pct.iloc[0])
        new_country_pct = 0.01

        if abs(old_country_pct - new_country_pct) < 1e-9:
            continue

        # scale all OTHER countries proportionally
        scale_factor = (1 - new_country_pct) / (1 - old_country_pct)

        group_mask        = (df['Channel ID'] == channel) & (df['w/c'] == wc)
        country_mask      = group_mask & (df['PlaceID'] == country_code)
        non_country_mask  = group_mask & (df['PlaceID'] != country_code)

        df.loc[non_country_mask, 'country_%'] *= scale_factor
        df.loc[country_mask, 'country_%'] = new_country_pct

    return df


# Iran = PlaceID "IRN"
IRAN_CODE = "IRN"

ig_country_df = fix_country_one_percent(
    ig_country_df,
    IRAN_CODE,
    ["2026-01-12"]
)
'''


# Iran = PlaceID "IRN"
IRAN_CODE = "IRN"

ig_country_df = fix_country_one_percent_prev_week(
    ig_country_df,
    IRAN_CODE,
    ["2026-01-19","2026-01-12"]
)


# In[13]:


#sanity checks 
# 1
#display(ig_country_df[(ig_country_df['w/c'] == '2026-01-12') & (ig_country_df['PlaceID'] == 'IRN')].head())
# 2
#display(ig_country_df[(ig_country_df['w/c'] == '2026-01-12') & (ig_country_df['Channel ID'] == 'INS17841402172935746')].sort_values('PlaceID'))
# 3 
print(ig_country_df[(ig_country_df['w/c'] == '2026-01-12') & (ig_country_df['Channel ID'] == 'INS17841402172935746')].sort_values('PlaceID')['country_%'].sum())

#sanity checks - previous weeks are not affected
# 4
#display(ig_country_df[(ig_country_df['w/c'] == '2026-01-05') & (ig_country_df['PlaceID'] == 'IRN')].head())
# 5
#display(ig_country_df[(ig_country_df['w/c'] == '2026-01-05') & (ig_country_df['Channel ID'] == 'INS17841402172935746')].sort_values('PlaceID'))
# 6 
print(ig_country_df[(ig_country_df['w/c'] == '2026-01-05') & (ig_country_df['Channel ID'] == 'INS17841402172935746')].sort_values('PlaceID')['country_%'].sum())


# In[14]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
ig_country_df[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv", 
                           index=None)
'''
compare_or_update_reference(ig_country_df[cols], 
                            f"../test/refactoring/{platformID}_country_expected.pkl", 
                            cols, update=False)
'''


# In[ ]:





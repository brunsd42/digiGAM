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
from functions import lookup_loader, execute_sql_query, calculate_rolling_avg_country_split, apply_first_split_backfill, compare_or_update_reference
import test_functions

from config import gam_info


# In[4]:


lookup = lookup_loader(gam_info, platformID, '1c',
                       with_country=True, country_col=['YT-_FBE_codes'])
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']
country_codes = lookup['country_codes']


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

try: 
    df = execute_sql_query(sql_query)
    df['page_id'] = platformID+facebook_country_raw['page_id']
    df.to_csv(file, index=False, na_rep='')
except:
    print("no redshift connection using last time queried")

facebook_country_raw = pd.read_csv(file, keep_default_na=False, dtype={"page_id": "string"}).drop_duplicates()
facebook_country_raw['week_commencing'] = pd.to_datetime(facebook_country_raw['week_commencing'])
facebook_country_raw = facebook_country_raw.rename(columns={'page_id': 'Channel ID',
                                                            'page_name': 'Channel Name',
                                                            'week_commencing': 'w/c',
                                                            'fb_metric_breakdown': 'YT-_FBE_codes'})



# In[7]:


channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()

### RUN TESTS
# missing page_ids
test_functions.test_filter_elements_returned(facebook_country_raw, 
                                             channel_ids, 
                                             'Channel ID', 
                                             f"{platformID}_1c_09",
                                             "Check that all page IDs are found in SQL")


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col='Channel ID',
                                               main_data=facebook_country_raw,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts[['Channel ID', 'Start', 'End']],
                                               test_number=f"{platformID}_1c_10",
                                               test_step="Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(facebook_country_raw, 
                           numeric_columns=['page_fans_country'], 
                           test_number=f"{platformID}_1c_11",
                           test_step='Check no missing values in page fans column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(facebook_country_raw, ['Channel ID', 'w/c', 'YT-_FBE_codes'], 
                               test_number=f"{platformID}_1c_12",
                               test_step='Check no duplicates from redshift returned')


# In[8]:


# filter to relevant channel ids
facebook_country = facebook_country_raw[facebook_country_raw['Channel ID'].isin(channel_ids)]

# fill missing countries 
facebook_country['YT-_FBE_codes'] = facebook_country['YT-_FBE_codes'].replace('', 'ZZ')

# filter to relevant countries
test_functions.test_inner_join(facebook_country, 
                               country_codes, 
                               ['YT-_FBE_codes'], 
                               f"{platformID}_1c_13",
                               test_step='calculating country %',
                                focus='left')

facebook_country = facebook_country.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], 
                                          on='YT-_FBE_codes', 
                                          how='left').drop(columns=['YT-_FBE_codes'])


# In[9]:


# Group by specified columns and sum the fb_metric_value
facebook_country_sum = facebook_country.groupby(['Channel ID', 'w/c'])\
                                        .agg(Sum_page_fans_country=('page_fans_country', 'sum'))\
                                        .reset_index()
facebook_country = facebook_country.merge(facebook_country_sum, 
                                          how='inner', on= ['Channel ID', 'w/c'])
test_functions.test_inner_join(facebook_country, facebook_country_sum, 
                               ['Channel ID', 'w/c'], 
                               f"{platformID}_1c_14", 
                               test_step='calculating country %')

facebook_country['country_%'] = facebook_country['page_fans_country']/facebook_country['Sum_page_fans_country']

### RUN TESTS
test_functions.test_percentage(facebook_country, 
                               ['Channel ID', 'w/c'], 
                               f"{platformID}_1c_15", 
                               test_step='summing country % per week & account')


# In[10]:


# calculate rolling average for missing weeks ?
avg_country_df = calculate_rolling_avg_country_split(facebook_country, 'country_%', 
                                                     week_tester['w/c'].min(), week_tester['w/c'].max())

# new channels have missing country splits for the first few weeks
avg_backfill_country_df = apply_first_split_backfill(avg_country_df, 
                                                          socialmedia_accounts, 
                                                          week_tester
                                                         )


# In[11]:


# Canonical channel list and canonical week list
channel_list = facebook_country[['Channel ID']].drop_duplicates()
week_list = week_tester[['w/c']].drop_duplicates()

# Build full (Channel × Week) grid via cross join
expected_grid = channel_list.merge(week_list, how='cross')

# Join channel start/end info and keep only weeks within each channel's active window
# - Include weeks on/after Start
# - Include weeks on/before End (or 'today' if End is missing)
expected_grid = expected_grid.merge(
    socialmedia_accounts[['Channel ID', 'Start', 'End']],
    on='Channel ID',
    how='left'
)

today = pd.Timestamp.today().normalize()
expected_grid = expected_grid[
    (expected_grid['Start'] <= expected_grid['w/c']) &
    (expected_grid['End'].fillna(today) >= expected_grid['w/c'])
]

# Left-join actual engagements to the expected grid to detect missing rows
filled_grid = expected_grid.merge(
    facebook_country,
    on=['Channel ID', 'w/c'],
    how='left',
    indicator=True
)

# Mark rows that are missing in the original data
filled_grid['missing_week'] = (filled_grid['_merge'] == 'left_only')
filled_grid = filled_grid.drop(columns=['_merge'])

# Join per-channel rolling averages (already computed) to supply fill values
# Suffix `_avg` ensures we can distinguish average columns from originals
result_df = filled_grid.merge(
    avg_backfill_country_df,
    on=['Channel ID', 'w/c'],
    how='left',
    suffixes=['', '_avg']
).drop_duplicates()

# Fill NaNs in original columns with per-channel averages (only where original is missing)
result_df['PlaceID']   = result_df['PlaceID'].fillna(result_df['PlaceID_avg'])
result_df['country_%'] = result_df['country_%'].fillna(result_df['country_%_avg'])

# duplicates because there is a row expansion between main dataset and average for weeks that have country data
cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
result_df = result_df[cols].drop_duplicates().dropna(subset='country_%')
# result_df now contains:
# - All (Channel ID, w/c) rows within each channel's Start–End window
# - `missing_week=True` for synthetic rows added from the expected grid
# - Filled `PlaceID` / `country_%` from per-channel averages where originals were NaN


# In[13]:


# missing weeks per page_id
test_functions.test_weeks_presence_per_account(key='w/c',
                                               channel_id_col='Channel ID',
                                               main_data=result_df,
                                               week_lookup=week_tester[['w/c']],
                                               channel_lookup=socialmedia_accounts[['Channel ID', 'Start', 'End']],
                                               test_number=f"{platformID}_1c_16",
                                               test_step="After rolling average: Check all weeks present for each account")

# missing values per week / page id 
test_functions.test_non_null_and_positive(result_df, 
                           numeric_columns=['country_%'], 
                           test_number=f"{platformID}_1c_17",
                           test_step='After rolling average: Check no missing values in page fans column from redshift returned')

# test for duplicate entries 
test_functions.test_duplicates(result_df, 
                               ['Channel ID', 'w/c', 'PlaceID'], 
                               test_number=f"{platformID}_1c_18",
                               test_step='After rolling average: Check no duplicates from redshift returned')


# In[14]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['Channel ID', 'Channel Name', 'w/c', 'PlaceID', 'country_%']
result_df[cols].to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv",
                        index=None)


# In[ ]:





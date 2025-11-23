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


# In[4]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
socialmedia_accounts = pd.read_excel("../helper/ins_account_lookup.xlsx")
channel_ids = socialmedia_accounts['Channel ID'].tolist()


# # ingestion

# # activity

# In[18]:


metric_ids = ['saved', 
              'shares', 
              'total_interactions', 
              'comments', 
              #'engagement',
              'impressions', 
              'likes', 
              'reach', 
              'plays', 
              #'video_views'
             ]

# content level / post level
sql_query = f""" 
    SELECT
        p.week_commencing AS w/c,
        p.account_id,
        p.page_id AS ig_user_id,
        p.page_name,
        GREATEST(0, SUM(COALESCE(p.engagements_week_diff, p.engagements))) AS engagements,
        GREATEST(0, SUM(COALESCE(p.comments_week_diff, p.comments))) AS comments,
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
    LEFT JOIN
        central_insights.adverity_social_instagram_by_reach AS r
        ON p.week_commencing = r.week_commencing
        AND p.account_id = r.account_id
    WHERE
        p.week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
    GROUP BY
        p.week_commencing,
        p.account_id,
        p.page_id,
        p.page_name;
        """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_activity_redshift_extract.csv"
df = execute_sql_query(sql_query)

df['ig_user_id'] = df['ig_user_id'].astype(str) 
df['ig_media_id'] = df['ig_media_id'].astype(str) 
df.to_csv(file, index=False, na_rep='')


# In[20]:


df


# In[ ]:


ig_views = pd.read_csv(file, keep_default_na=False)

ig_views['ig_user_id'] = ig_views['ig_user_id'].astype(str) 
ig_views['w/c'] = pd.to_datetime(ig_views['w/c'])
# Run the tests
column_name = 'ig_user_id'
test_functions.test_filter_elements_returned(ig_views, channel_ids, column_name, 
                                             "1_IG_1", "ig_media_insights sql query - user test")
column_name = 'ig_metric_id'
test_functions.test_filter_elements_returned(ig_views, metric_ids, column_name, 
                                             "1_IG_2", "ig_media_insights sql query - metric test")

test_functions.test_weeks_presence('week_ending', 
                                    ig_views.rename(columns={'ig_metric_end_time': 'week_ending'}), 
                                    week_tester, '1_IG_3', "ig_media_insights sql query")



# # metadata

# In[ ]:


sql_query = f""" 
    SELECT 
        ig_user_id, 
        ig_user_name, 
        ig_media_id, 
        ig_media_type, 
        ig_media_product_type, 
        week_ending, 
        ig_media_created_time, 
        ig_user_bbc_bus_unit
    FROM 
        redshiftdb.central_insights.ig_media_metadata
    WHERE
        week_ending BETWEEN '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}'
    ;
    """

file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_metadata_redshift_extract.csv"
'''df = execute_sql_query(sql_query)
df['ig_user_id'] = df['ig_user_id'].astype(str) 
df['ig_media_id'] = df['ig_media_id'].astype(str) 

df.to_csv(file, index=False, na_rep='')
'''
ig_metadata = pd.read_csv(file, keep_default_na=False)

ig_metadata['ig_user_id'] = ig_metadata['ig_user_id'].astype(str) 
ig_metadata['ig_media_id'] = ig_metadata['ig_media_id'].astype(str) 
ig_metadata['week_ending'] = pd.to_datetime(ig_metadata['week_ending'])
# Run the tests
column_name = 'ig_user_id'
test_functions.test_filter_elements_returned(ig_metadata, channel_ids, column_name, 
                                             "1_IG_4", "ig_media_metadata sql query - channel test")

test_functions.test_weeks_presence('week_ending', ig_metadata, week_tester, 
                                   '1_IG_5', "ig_media_insights sql query")

ig_metadata.sample()


# # combine media metrics and their metadata

# In[ ]:


ig_views = ig_views.rename(columns={'ig_metric_end_time': 'week_ending'})
ig_views['week_ending'] = pd.to_datetime(ig_views['week_ending'])
ig_metadata['week_ending'] = pd.to_datetime(ig_metadata['week_ending'])
left_cols = ['ig_media_id', 'ig_metric_id', 'ig_metric_period', 'week_ending', 'ig_metric_breakdown', 'ig_metric_value']

ig_combine = ig_views[left_cols].merge(ig_metadata, how='inner', on=['ig_media_id', 'week_ending'])
test_functions.test_inner_join(ig_views, ig_metadata, ['ig_media_id', 'week_ending'],
                               '1_IG_6', test_step="combining ig_media_insights & ig_media_metadata ")

ig_combine.columns


# ## processing

# In[ ]:


# to get to account level 
# summarised are these columns: 
#     'ig_media_id', 'ig_media_type', 'ig_media_product_type', 'ig_media_created_time', 
ig_media_to_user = ig_combine.groupby(['ig_user_id', 'ig_user_name', 'ig_user_bbc_bus_unit', 
                                 'week_ending',
                                 'ig_metric_id', 'ig_metric_period', 'ig_metric_breakdown'])['ig_metric_value'].sum().reset_index()

# Rename the column for sum
ig_media_to_user.rename(columns={'ig_metric_value': 'Sum_ig_metric_value'}, inplace=True)

# Perform crosstab operation
ig_media_to_user = pd.pivot_table(ig_media_to_user, 
                             values='Sum_ig_metric_value', 
                             index=['ig_user_id', 'ig_user_name', 'ig_user_bbc_bus_unit', 
                                    'week_ending', 
                                    'ig_metric_period', 'ig_metric_breakdown'],
                             columns='ig_metric_id', 
                             aggfunc='sum').reset_index()

# test there is now one row per channel / week
test_functions.test_duplicates(ig_media_to_user, ['ig_user_id', 'week_ending'], 
                               "1_IG_7", "bringing IG media level to account level")



# ## user insights

# In[ ]:


# account level (would also contain content level but we don't what this here wil be replaced )

metric_ids = ['impressions', 'reach']
formatted_metric_ids = ', '.join(f"'{metric_id}'" for metric_id in metric_ids)

sql_query = f""" 
    SELECT 
        ig_user_id, 
        ig_user_name, 
        ig_metric_id, 
        ig_metric_period, 
        ig_metric_breakdown, 
        ig_metric_end_time, 
        ig_metric_value
    FROM 
        redshiftdb.central_insights.ig_user_insights
    WHERE
         (
         ig_metric_id in ({formatted_metric_ids}) 
         AND
         ig_metric_end_time BETWEEN '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}')
    ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_userInsights_redshift_extract.csv"
'''df = execute_sql_query(sql_query)
df['ig_user_id'] = df['ig_user_id'].astype(str) 
df.to_csv(file, index=False, na_rep='')
'''
ig_userInsights = pd.read_csv(file, keep_default_na=False)
ig_userInsights['ig_user_id'] = ig_userInsights['ig_user_id'].astype(str) 
ig_userInsights['ig_metric_end_time'] = pd.to_datetime(ig_userInsights['ig_metric_end_time'])
# Run the tests
column_name = 'ig_user_id'
test_functions.test_filter_elements_returned(ig_userInsights, channel_ids, column_name, "1_IG_8", "ig_user_insights sql query - user test")

column_name = 'ig_metric_id'
test_functions.test_filter_elements_returned(ig_userInsights, metric_ids, column_name, "1_IG_9", "ig_user_insights sql query - metric test")

test_functions.test_weeks_presence('week_ending', ig_userInsights.rename(columns={'ig_metric_end_time': 'week_ending'}), 
                                   week_tester, '1_IG_10', "ig_user_insights sql query")


# In[ ]:


# Pivot the DataFrame
ig_user_by_userInsights = ig_userInsights.pivot_table(index=['ig_user_id', 'ig_user_name', 'ig_metric_period',
                                                             'ig_metric_breakdown', 'ig_metric_end_time'],
                                                      columns='ig_metric_id',
                                                      values='ig_metric_value',
                                                      aggfunc='sum').reset_index()

# Flatten the column headers
ig_user_by_userInsights.columns.name = None
ig_user_by_userInsights.columns = [col if isinstance(col, str) else col[1] for col in ig_user_by_userInsights.columns]
#ig_user_by_userInsights.sample()
ig_user_by_userInsights = ig_user_by_userInsights.rename(columns={'ig_metric_end_time': 'week_ending'})
ig_user_by_userInsights['week_ending'] = pd.to_datetime(ig_user_by_userInsights['week_ending'])

# test there is now one row per channel / week
test_functions.test_duplicates(ig_user_by_userInsights, ['ig_user_id', 'week_ending'], 
                               "1_IG_11", "get unique user entries per channel & week")

'''
ig_user_by_userInsights_allWeeks = joining_allWeeks_perChannel(ig_user_by_userInsights, 'week_ending', 'ig_user_id',
                                                  week_tester, 
                                                  ['ig_user_id', 'ig_user_name', 'ig_metric_period', 'ig_metric_breakdown']
                                                              )'''


# # combine 

# In[ ]:


# excluded from left 'impressions', 'reach'
left_cols = ['ig_user_id', 'ig_user_name', 'ig_user_bbc_bus_unit', 
             'week_ending','ig_metric_period', 'ig_metric_breakdown',
             'comments', 'likes', 'plays',
             'saved', 'shares', 'total_interactions', #'video_views', 'engagement'
            ] 
# excluded from right #'ig_user_name', 'ig_metric_period', 'ig_metric_breakdown',
right_cols = ['ig_user_id', 'week_ending', 'impressions', 'reach']
ig_user_by_allMetrics = ig_media_to_user[left_cols].merge(ig_user_by_userInsights[right_cols], 
                                                                how='outer', #indicator=True,
                                                                on=['ig_user_id', 'week_ending']
                                                               )
test_functions.test_inner_join(ig_media_to_user[left_cols], 
                               ig_user_by_userInsights[right_cols], 
                               ['ig_user_id', 'week_ending'],
                               "1_IG_12", test_step="joining user insights to media insights")


# # user metadata redshift query 

# In[ ]:


sql_query = f""" 
    SELECT 
        ig_user_id, 
        ig_user_ig_id, 
        ig_user_name, 
        ig_user_username, 
        ig_user_linked_fb_page_id, 
        ig_user_bbc_category, 
        ig_user_bbc_nation, 
        ig_user_bbc_clean_name, 
        ig_user_bbc_bus_unit, 
        week_ending
    FROM 
        redshiftdb.central_insights.ig_user_metadata
    WHERE
        (
        week_ending BETWEEN '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}')
    ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_userMetadata_redshift_extract.csv"

'''df = execute_sql_query(sql_query)

df['ig_user_id'] = df['ig_user_id'].astype(str) 
df['ig_user_ig_id'] = df['ig_user_ig_id'].astype(str) 
df['ig_user_linked_fb_page_id'] = df['ig_user_linked_fb_page_id'].astype(str) 

df.to_csv(file, index=False, na_rep='')
'''
ig_userMetadata = pd.read_csv(file, keep_default_na=False)

ig_userMetadata['ig_user_id'] = ig_userMetadata['ig_user_id'].astype(str) 
ig_userMetadata['ig_user_ig_id'] = ig_userMetadata['ig_user_ig_id'].astype(str) 
ig_userMetadata['ig_user_linked_fb_page_id'] = ig_userMetadata['ig_user_linked_fb_page_id'].astype(str) 
ig_userMetadata['week_ending'] = pd.to_datetime(ig_userMetadata['week_ending'])
# Run the tests
column_name = 'ig_user_id'
test_functions.test_filter_elements_returned(ig_userMetadata, channel_ids, 
                                             column_name, "1_IG_13", "ig_user_metadata sql query - user test")

test_functions.test_weeks_presence('week_ending', ig_userMetadata, 
                                   week_tester, '1_IG_14', "ig_user_insights sql query")

ig_userMetadata.sample()


# In[ ]:


# just a clean list of the overview columns 
# groupby
cols = ['ig_user_id', 'ig_user_ig_id', 'ig_user_name', 'ig_user_username',
        'ig_user_linked_fb_page_id', 'ig_user_bbc_category', 'ig_user_bbc_nation', 
        'ig_user_bbc_clean_name', 'ig_user_bbc_bus_unit']

ig_userMetadata_slim = ig_userMetadata[cols].drop_duplicates()
ig_userMetadata_slim.shape


# In[ ]:


# combine all

# left out 'ig_user_name', 'ig_user_bbc_bus_unit'
right_cols = ['ig_user_id', 'ig_user_ig_id',  'ig_user_username',
       'ig_user_linked_fb_page_id', 'ig_user_bbc_category',
       'ig_user_bbc_nation', 'ig_user_bbc_clean_name', ]
ig_combine_final = ig_user_by_allMetrics.merge(ig_userMetadata_slim[right_cols], how='inner', 
                                               on=['ig_user_id'])
'''test_functions.test_inner_join(ig_user_by_allMetrics, ig_userMetadata_slim[right_cols], 
                               'ig_user_id', '1_IG_15', 'combine user metadata to rest')
'''
ig_combine_final.to_csv(f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv", 
                          index=None)


# ### milestone II

# In[ ]:


ig_engagements = ig_combine_final.merge(week_tester[['week_ending', 'w/c']], on='week_ending', how='left')
print(ig_engagements.shape)
ig_engagements = ig_engagements[ig_engagements['w/c'] >= "2024-04-01"]
print(ig_engagements.shape)


cols = ['w/c', 'ig_user_id', 'ig_user_name', 'ig_user_bbc_bus_unit', 
       'ig_metric_period', 'ig_metric_breakdown', 
        'comments', 'likes', 'plays', 'saved', 'shares', 'total_interactions', 'impressions', 
        'reach', 
        'ig_user_ig_id', 'ig_user_username', 'ig_user_linked_fb_page_id',
       'ig_user_bbc_category', 'ig_user_bbc_nation', 'ig_user_bbc_clean_name',
       ]
ig_engagements['total_interactions'] = ig_engagements['total_interactions'].fillna(0)
ig_engagements['comments'] = ig_engagements['comments'].fillna(0)
ig_engagements['likes'] = ig_engagements['likes'].fillna(0)
ig_engagements['shares'] = ig_engagements['shares'].fillna(0)
ig_engagements['saved'] = ig_engagements['saved'].fillna(0)

ig_engagements['weekly_media_engagements'] = ig_engagements[cols].apply(
    lambda row: row['total_interactions'] 
    if row['total_interactions'] > (row['comments'] + row['likes'] + row['shares'] + row['saved']) 
    else (row['comments'] + row['likes'] + row['shares'] + row['saved']), axis=1
)
ig_engagements['daily_avg_reach'] = 0

cols = ['Channel ID', 'IG FB Linked ID', 'Channel Name', 'IG Account URL',	
         'IG studios exc uk', 'ServiceID', 'ServiceName',]
ig_engagements = ig_engagements.rename(columns={"ig_user_name":"Channel Name",
                                                "ig_user_id": "Channel ID"})\
                                .merge(socialmedia_accounts[cols], on='Channel Name', how='left', 
                                       indicator=True)
ig_engagements['Channel ID'] = ig_engagements['Channel ID_x'].fillna('Channel ID_y')
ig_engagements = ig_engagements.drop(columns=['Channel ID_x', 'Channel ID_y'])


# In[ ]:


ig_engagements_name_matched = ig_engagements[ig_engagements['_merge'] == 'both']
ig_engagements_unmatched = ig_engagements[ig_engagements['_merge'] == 'left_only']

ig_engagements_id_matched = ig_engagements_unmatched.merge(socialmedia_accounts[cols], 
                                                           on='Channel ID', how='inner')
cols_to_clean = ['IG FB Linked ID', 'Channel Name', 'IG studios exc uk', 'IG Account URL',
                 'ServiceID', 'ServiceName',]
for col in cols_to_clean:
    ig_engagements_id_matched[f"{col}"] = ig_engagements_id_matched[f"{col}_x"].fillna(ig_engagements_id_matched[f"{col}_y"])
    ig_engagements_id_matched = ig_engagements_id_matched.drop(columns=[f"{col}_x", f"{col}_y"])

ig_engagements_final = pd.concat([ig_engagements_name_matched, ig_engagements_id_matched])

ig_engagements_final['ig_user_linked_fb_page_id'] = ig_engagements_final['ig_user_linked_fb_page_id'].fillna(ig_engagements_final['IG FB Linked ID'])
ig_engagements_final = ig_engagements_final.drop(columns=['IG FB Linked ID'])

cols = ['Channel ID', 'Channel Name', 'ig_user_linked_fb_page_id', 'IG Account URL', 
        'w/c', 'ServiceID', 'IG studios exc uk', 
        'weekly_media_engagements', 'saved', 'plays', 'daily_avg_reach', 'reach', 'impressions']

ig_engagements_final = ig_engagements_final[cols]
ig_engagements_final.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv", 
                          index=None)


# ### Milestone III

# In[ ]:


# Define the multiplier mapping
service_multipliers = {
    "AZE": 0.79,
    "INO": 0.71,
    "SPA": 0.76,
    "RUS": 0.68,
    "UKR": 0.76,
    "URD": 0.81
}

ig_engagements_final['plays'] = ig_engagements_final['plays'].fillna(0)
ig_engagements_final['impressions'] = ig_engagements_final['impressions'].fillna(0)
ig_engagements_final['reach'] = ig_engagements_final['reach'].fillna(0)

# Apply the logic to create a new column 'adjusted_reels_plays'
ig_engagements_final['30 view'] = ig_engagements_final.apply(
    lambda row: row['plays'] * service_multipliers.get(row['ServiceID'], 0.668476004281482),
    axis=1
)

def safe_ratio(row):
    if row['reach'] == 0 or pd.isna(row['reach']):
        return None  # or np.nan if you prefer
    if row['ServiceID'] == 'PER':
        return (row['impressions'] / row['reach']) * 0.052990833 + 1.693711126
    else:
        return (row['impressions'] / row['reach']) * 0.21907280062318 + 1.20241835848198

ig_engagements_final['IG Modelled Factor'] = ig_engagements_final.apply(safe_ratio, axis=1)

def compute_engagement_estimate(row):
    if row['ServiceID'] == 'PER':
        if row['IG Account URL'] == 'instagram.com/bbcpersian':
            factor = row.get('IG Modelled Factor', None)
            if factor and factor != 0:
                return (row['weekly_media_engagements'] + row['30 view']) / factor
            else:
                return None
        else:
            return row.get('IG Engaged Users', None)
    else:
        factor = row.get('IG Modelled Factor', None)
        if factor and factor != 0:
            estimate = (row['weekly_media_engagements'] + row['30 view']) / factor
            return min(estimate, row['reach'])
        else:
            return None

ig_engagements_final['IG Engaged Users'] = ig_engagements_final.apply(compute_engagement_estimate, axis=1)


def compute_persian_engagement(row):
    if row['IG Account URL'] == 'instagram.com/bbcpersian':
        factor = row.get('IG Modelled Factor', None)
        if factor and factor != 0:
            return (row['weekly_media_engagements'] + row['30 view']) / factor
        else:
            return None
    else:
        return row.get('IG Engaged Users', None)

ig_engagements_final['IG Engaged Persian Exception'] = ig_engagements_final.apply(compute_persian_engagement, axis=1)
ig_engagements_final['IG Engaged Persian Exception'] = ig_engagements_final.apply(
    lambda row: row['reach'] if row['IG Engaged Persian Exception'] > row['reach'] 
                             else row['IG Engaged Persian Exception'],
    axis=1
)
ig_engagements_final.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_engagements_final.csv", 
                          index=None)


# In[ ]:


ig_engagements_final[(ig_engagements_final['Channel ID'] =='17841401606325704')]


# In[ ]:





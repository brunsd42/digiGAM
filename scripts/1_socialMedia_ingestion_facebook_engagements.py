#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
import numpy as np


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


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
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


# # engagements 

# In[5]:


metric_ids = [
    #'page_engaged_users', 
    #'page_consumptions', 
    'page_post_engagements',
    'page_video_views_autoplayed', 
    'page_video_complete_views_30s_autoplayed', 
    'page_video_complete_views_30s', 
    'page_video_complete_views_30s_unique', 
    'page_consumptions_by_consumption_type',
    'page_consumptions_by_consumption_type_unique',
    'page_consumptions_unique', 
    'page_impressions', 
    'page_impressions_unique',
    'page_video_views_10s_autoplayed', 
    'page_video_views_10s', 
    'page_video_views_10s_unique',
    
]
formatted_metric_ids = ', '.join(f"'{metric_id}'" for metric_id in metric_ids)


sql_query = f"""
    SELECT 
        fb_page_id, 
        fb_page_name, 
        fb_metric_id, 
        fb_metric_period, 
        fb_metric_breakdown, 
        fb_metric_end_time, 
        fb_metric_value
    FROM
        redshiftdb.central_insights.fb_page_insights
    WHERE
        fb_metric_id in ({formatted_metric_ids})
        AND 
        fb_metric_end_time Between  '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}'
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_engagements_redshift_extract.csv"

df = execute_sql_query(sql_query)
df['fb_page_id'] = df['fb_page_id'].astype(str)

df.to_csv(file, index=False, na_rep='')

#facebook_engagements_raw = pd.read_csv(f"../data/raw/Facebook/{gam_info['file_timeinfo']}_facebook_engagements_redshift_extract.csv")
facebook_engagements_raw = pd.read_csv(file, keep_default_na=False)
facebook_engagements_raw['fb_page_id'] = facebook_engagements_raw['fb_page_id'].astype(str)
facebook_engagements_raw['fb_metric_end_time'] = pd.to_datetime(facebook_engagements_raw['fb_metric_end_time'])
print(facebook_engagements_raw.shape)

# Run the tests
column_name = 'fb_page_id'
test_functions.test_filter_elements_returned(facebook_engagements_raw, channel_ids, 
                                             column_name, "1_FB_1", "engagement sql query - page test")

column_name = 'fb_metric_id'
test_functions.test_filter_elements_returned(facebook_engagements_raw, metric_ids, 
                                             column_name, "1_FB_2", "engagement sql query - metric test")
 
test_functions.test_weeks_presence('week_ending', 
                                    facebook_engagements_raw.rename(columns={'fb_metric_end_time': 'week_ending'}), 
                                    week_tester, '1_FB_3', "engagement sql query")


# In[ ]:





# ## processing engagements

# In[7]:


subset_cols = ['page_consumptions_by_consumption_type', 
               'page_consumptions_by_consumption_type_unique']
subset_consumptionType = facebook_engagements_raw[facebook_engagements_raw['fb_metric_id'].isin(subset_cols)]
print(f"unique metric breakdowns: {subset_consumptionType.fb_metric_breakdown.unique()}")
# Group by specified columns and sum the fb_metric_value
subset_consumptionType_grouped = subset_consumptionType.groupby(['fb_page_id', 'fb_page_name', 'fb_metric_end_time',
                                                                 'fb_metric_period', 'fb_metric_id'])\
                                                       .agg({'fb_metric_value': 'sum'}).reset_index()

subset_nonConsumptionType = facebook_engagements_raw[~facebook_engagements_raw['fb_metric_id'].isin(subset_cols)]
subset_nonConsumptionType = subset_nonConsumptionType.drop(columns='fb_metric_breakdown') #because this column is None for the whole dataframe

facebook_engagements_raw = pd.pivot_table(pd.concat([subset_consumptionType_grouped, subset_nonConsumptionType]), 
                             values='fb_metric_value', 
                             index=['fb_page_id', 'fb_page_name', 
                                    'fb_metric_period', 'fb_metric_end_time' ], 
                             columns='fb_metric_id', 
                             aggfunc='sum').reset_index()

facebook_engagements_raw.to_csv(f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                           index=None, na_rep='')


# In[8]:


''' this gets filtered out in the next workflow
# consumptions combination of the various types:
grouped_df = facebook_engagements_raw.groupby(['fb_page_id', 'fb_metric_end_time', 'fb_page_name', 'fb_metric_period'])\
                                 .agg(page_consumptions=('page_consumptions_by_consumption_type', 'sum'))\
                                 .reset_index()
cols_right = ['fb_page_id', 'fb_metric_end_time', 'page_consumptions']
facebook_engagements_recalc = facebook_engagements_raw.merge(grouped_df[cols_right], how='inner',
                                                  on=['fb_page_id', 'fb_metric_end_time'])

test_functions.test_inner_join(facebook_engagements_raw, grouped_df[cols_right], 
                               ['fb_page_id', 'fb_metric_end_time'], 
                               '2_FB_1', 'adding page_consumptions')
facebook_engagements_recalc.sample()'''


# In[9]:


metric_col = ['page_post_engagements',
              'page_consumptions_by_consumption_type'
              ]
for col in metric_col:
    facebook_engagements_raw[col] = facebook_engagements_raw[col].fillna(0)
    
facebook_engagements_raw['page_consumptions_by_consumption_type'] = np.where(
    facebook_engagements_raw['page_consumptions_by_consumption_type'] == 0,
    facebook_engagements_raw['page_post_engagements'],
    facebook_engagements_raw['page_consumptions_by_consumption_type']
)


# In[10]:


# adding week info
facebook_engagements_raw = facebook_engagements_raw.rename(columns={'fb_metric_end_time': 'week_ending'})
facebook_engagements_raw['week_ending'] = pd.to_datetime(facebook_engagements_raw['week_ending'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])
cols_right = ["week_ending" , "WeekNumber_finYear", "w/c"]
facebook_engagements_recalc = facebook_engagements_raw.merge(week_tester[cols_right], how='inner',
                                                                               on='week_ending' )

facebook_engagements_recalc = facebook_engagements_recalc.drop(columns=['week_ending'])
ma_emplifi = pd.read_excel("../data/interim/Facebook_MediaAction_GAM2025_Processed.xlsx")
ma_emplifi['fb_page_id'] = ma_emplifi['fb_page_id'].astype(str)
ma_emplifi['w/c'] = pd.to_datetime(ma_emplifi['w/c']).dt.strftime('%Y-%m-%d')

########
# Define the key columns
keys = ['fb_page_id', 'w/c']

# Ensure both DataFrames have these columns as strings for consistency
facebook_engagements_recalc[keys] = facebook_engagements_recalc[keys].astype(str)
ma_emplifi[keys] = ma_emplifi[keys].astype(str)

# Find rows in ma_emplifi that are NOT in facebook_engagements_recalc
mask = ~ma_emplifi.set_index(keys).index.isin(facebook_engagements_recalc.set_index(keys).index)
ma_subset = ma_emplifi[mask]

# Now concatenate only the missing rows
facebook_engagements_recalc = pd.concat([facebook_engagements_recalc, ma_subset], ignore_index=True)


# In[13]:


wse_emplifi = pd.read_excel("../data/interim/GAM2025_WSE_FBE.xlsx")
wse_emplifi['fb_page_id'] = wse_emplifi['fb_page_id'].astype(str)

wse_emplifi = wse_emplifi.rename(columns={'fb_metric_end_time': 'week_ending'})
wse_emplifi['week_ending'] = pd.to_datetime(facebook_engagements_raw['week_ending'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])
cols_right = ["week_ending" , "WeekNumber_finYear", "w/c"]
wse_emplifi = wse_emplifi.merge(week_tester[cols_right], how='inner', on='week_ending' )
wse_emplifi['w/c'] = pd.to_datetime(wse_emplifi['w/c']).dt.strftime('%Y-%m-%d')

########
# Define the key columns
keys = ['fb_page_id', 'w/c']

# Ensure both DataFrames have these columns as strings for consistency
facebook_engagements_recalc[keys] = facebook_engagements_recalc[keys].astype(str)
wse_emplifi[keys] = wse_emplifi[keys].astype(str)

# Find rows in ma_emplifi that are NOT in facebook_engagements_recalc
mask = ~wse_emplifi.set_index(keys).index.isin(facebook_engagements_recalc.set_index(keys).index)
wse_subset = wse_emplifi[mask]

# Now concatenate only the missing rows
facebook_engagements_recalc = pd.concat([facebook_engagements_recalc, wse_subset], ignore_index=True)



# In[14]:


facebook_engagements_recalc = facebook_engagements_recalc.rename(columns={
    'page_consumptions_by_consumption_type': 'page_consumptions'
})

metric_col = ['page_video_complete_views_30s_autoplayed',
              'page_video_complete_views_30s',
              'page_post_engagements',
              'page_consumptions',
              'page_consumptions_unique',
              'page_video_complete_views_30s_unique',
              'page_video_views_10s',
              'page_video_views_10s_unique',
              'page_video_views_10s_autoplayed',]
for col in metric_col:
    facebook_engagements_recalc[col] = facebook_engagements_recalc[col].fillna(0)
    
# join with GAM lookup table on w/c 
cols_left = ["fb_page_id", "fb_page_name", "fb_metric_period", 
             'WeekNumber_finYear', 'w/c', #"page_engaged_users", 
             "page_consumptions", "page_video_views_10s", 'page_video_views_10s_autoplayed',
             'page_video_views_10s_unique',
             "page_video_views_autoplayed", "page_video_complete_views_30s_autoplayed",
             "page_consumptions_unique", "page_impressions", "page_impressions_unique",
             "page_video_complete_views_30s", "page_video_complete_views_30s_unique",
             "page_consumptions_by_consumption_type", "page_consumptions_by_consumption_type_unique"
            ]

# account test
column_name = 'fb_page_id'
test_functions.test_filter_elements_returned(facebook_engagements_recalc, channel_ids, column_name,
                                            '2_FB_3', 'ensuring no weeks were lost during FB factors & page consumption calculation')

facebook_engagements_recalc = facebook_engagements_recalc.rename(columns={"fb_page_id": "Channel ID"})\
    .merge(socialmedia_accounts[['Channel ID', 'ServiceID', 'Excluding UK']],
           on='Channel ID', how='left')

facebook_engagements_recalc = facebook_engagements_recalc.sort_values(['Channel ID', 'w/c', 'page_video_views_10s'], ascending=True)
facebook_engagements = facebook_engagements_recalc.groupby(['Channel ID', 'w/c']).first().reset_index()

# autoplay start
autoplay = facebook_engagements.copy()
# Calculate 'Autoplay %' safely
autoplay['Autoplay %'] = (
    autoplay['page_video_complete_views_30s_autoplayed'] /
    autoplay['page_video_complete_views_30s']
).clip(upper=1)  # Ensures values don't exceed 1

autoplay_cols = ['Channel ID', 'fb_page_name', 'ServiceID',
                 'WeekNumber_finYear', 'w/c',
                 'Autoplay %', 'page_video_views_10s', 
                 'page_video_views_10s_autoplayed', 'page_video_views_10s_unique']

autoplay[autoplay_cols].to_csv(f"../data/interim/{gam_info['file_timeinfo']}_{platformID}_autoplay_proportion.csv",
                                                       index=None)
# autoplay done
rename = {'page_consumptions': 'Weekly Consumptions',
          'page_consumptions_unique': 'Weekly Engaged Users',
          'page_video_complete_views_30s_autoplayed': 'Weekly page_video_complete_views_30s_autoplayed',

         }
engagement_cols = ['ServiceID', #'Channel Group', 
                   'Channel ID', 'fb_page_name',
                   'WeekNumber_finYear', 'w/c',
                   'Weekly Consumptions', 'Weekly Engaged Users', 'Weekly page_video_complete_views_30s_autoplayed', 
                   'page_video_complete_views_30s', 'page_video_complete_views_30s_unique', 
                   'page_video_views_10s', 'page_video_views_10s_autoplayed', 'page_video_views_10s_unique',
                   'page_video_views_autoplayed', 
                  ]

facebook_engagements = facebook_engagements.rename(columns=rename)[engagement_cols]

facebook_engagements['Views per Viewer'] = facebook_engagements['page_video_complete_views_30s'] / facebook_engagements['page_video_complete_views_30s_unique']
facebook_engagements['10s Views per Viewer'] = facebook_engagements['page_video_views_10s'] / facebook_engagements['page_video_views_10s_unique']
facebook_engagements['autoplay 10s factor'] = facebook_engagements['page_video_views_10s_autoplayed'] / facebook_engagements['page_video_views_10s']

facebook_engagements['autoplay viewers'] = facebook_engagements['Weekly page_video_complete_views_30s_autoplayed'] / facebook_engagements['Views per Viewer']
facebook_engagements['autoplay viewers 10s'] = facebook_engagements['page_video_views_10s_unique'] / facebook_engagements['10s Views per Viewer']


# milestone II

# In[20]:


facebook_engagements[(facebook_engagements['Channel ID'] == '163571453661989') & 
    (facebook_engagements['ServiceID'] == 'WSE') & 
    (facebook_engagements['w/c'] == '2025-02-03')].sort_values(by='Channel ID').T


# In[16]:


user_engagements = pd.read_excel("../data/interim/FB Content Engagements per Engaged User - 2024.xlsx")
user_engagements['Avg_Engagements per Engaged User'] = user_engagements['Avg_Engagements per Engaged User']*1.0949389
metric_col = ['Avg_Engagements per Engaged User',
    ]
for col in metric_col:
    user_engagements[col] = user_engagements[col].fillna(0)

# possible error source! 
user_engagements = user_engagements.drop_duplicates(subset=['FB Page ID']).rename(columns={'FB Page ID': 'Channel ID'})
user_engagements['Channel ID'] = user_engagements['Channel ID'].apply(lambda x: str(int(x)))

facebook_engagements_user = facebook_engagements.merge(user_engagements, on='Channel ID', how='inner')

facebook_engagements_user['Weekly Engaged Users'] = np.where(facebook_engagements_user['Weekly Engaged Users']==0, 
                                                             facebook_engagements_user['Weekly Consumptions']/facebook_engagements_user['Avg_Engagements per Engaged User'], 
                                                             facebook_engagements_user['Weekly Engaged Users'])

facebook_engagements_user['Engaged Reach'] = np.where(facebook_engagements_user['Weekly Engaged Users']>facebook_engagements_user['autoplay viewers 10s'],
                                                         facebook_engagements_user['Weekly Engaged Users']+facebook_engagements_user['autoplay viewers 10s']*0.04827,
                                                         facebook_engagements_user['autoplay viewers 10s']+(facebook_engagements_user['Weekly Engaged Users']*0.04822))
facebook_engagements_user['FLAG'] = np.where(facebook_engagements_user['Engaged Reach']<facebook_engagements_user['autoplay viewers 10s'], "FLAG1", 
                                            np.where(facebook_engagements_user['Engaged Reach']<facebook_engagements_user['Weekly Engaged Users'], "FLAG2", ""))
print(facebook_engagements_user['FLAG'].value_counts())

cols = ['ServiceID', #'Channel Group',
        'page_video_views_10s', 'page_video_views_10s_autoplayed', 'page_video_views_10s_unique',
        'Channel ID', #'fb_page_name',
        'w/c', 'WeekNumber_finYear', 
        '10s Views per Viewer', 'autoplay 10s factor', 'autoplay viewers 10s',
        'Avg_Engagements per Engaged User', 'Engaged Reach', 
       ]

facebook_engagements_user[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                                                                   index=None)


# In[19]:


facebook_engagements_user[(facebook_engagements_user['Channel ID'] == '163571453661989') & 
    (facebook_engagements_user['ServiceID'] == 'WSE') & 
    (facebook_engagements_user['w/c'] == '2025-02-03')].sort_values(by='Channel ID').T


# ## creating click type dataset which is used for facebook factors

# In[18]:


subset_consumptionType.to_excel(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_click type.xlsx", index=None)



# In[ ]:





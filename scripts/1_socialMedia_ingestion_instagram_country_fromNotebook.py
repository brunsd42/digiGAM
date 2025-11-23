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

# social media accounts 
socialmedia_accounts = pd.read_excel(f"../helper/ins_account_lookup.xlsx")


# # ingestion

# In[5]:


sql_query = f"""
    SELECT 
        page_id,
        page_name,
        week_commencing,
        period,
        country_name,
        followers_by_demographic
    FROM
        redshiftdb.central_insights.adverity_social_instagram_by_page_demo
    WHERE
        week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_userCountry_redshift_extract.csv"

df = execute_sql_query(sql_query)
df['ig_user_id'] = df['ig_user_id'].astype(str) 

df.to_csv(file, index=False, na_rep='')

ig_userCountry = pd.read_csv(file, keep_default_na=False)
ig_userCountry['ig_user_id'] = ig_userCountry['ig_user_id'].astype(str) 

grouped_df = ig_userCountry.groupby(['ig_user_id', 'ig_user_name', 'ig_metric_id',
                                     'ig_metric_period', 'ig_metric_end_time']).agg({'ig_metric_value': 'sum'}).reset_index()

# Rename the aggregated column
grouped_df = grouped_df.rename(columns={'ig_metric_value': 'Sum_ig_metric_value'})
right_cols = ['ig_user_id', 'ig_metric_end_time', 'Sum_ig_metric_value']
ig_country_df_sum = ig_userCountry.merge(grouped_df[right_cols], 
                                          how='inner',
                                          on=['ig_user_id', 'ig_metric_end_time'])

test_functions.test_inner_join(ig_userCountry, grouped_df[right_cols], 
                               ['ig_user_id', 'ig_metric_end_time'],
                               '1_IG_21')

ig_country_df_sum['country_%'] = ig_country_df_sum['ig_metric_value'] / ig_country_df_sum['Sum_ig_metric_value']

ig_country_df_sum.to_csv(f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv", 
                           index=None)


# In[6]:


df


# In[ ]:


# week
print("... cleaning weeks ... ")
ig_country_df_sum['ig_metric_end_time'] = pd.to_datetime(ig_country_df_sum['ig_metric_end_time'])
ig_country_df = ig_country_df_sum.rename(columns={'ig_metric_end_time': 'week_ending'})\
                                  .merge(week_tester[['week_ending', 'w/c']], on='week_ending', 
                                         how='left', indicator=True)
print(ig_country_df._merge.value_counts())
ig_country_df = ig_country_df.drop(columns=['_merge'])
print(ig_country_df.shape)
ig_country_df = ig_country_df[ig_country_df['w/c'] >= "2024-04-01"]
print(ig_country_df.shape)

# social media accounts
# channel ID
print("... cleaning channels ... ")
ig_country_df = ig_country_df.rename(columns={'ig_user_id': 'Channel ID', 
                                              'ig_user_name': 'Channel Name'})
ig_country_df = ig_country_df.merge(socialmedia_accounts[['Channel ID', 'ServiceName']], 
                                    on='Channel ID', how='left', indicator=True)
ig_id_matched = ig_country_df[ig_country_df['_merge'] == 'both'].drop(columns=['_merge'])
ig_unmatched = ig_country_df[ig_country_df['_merge'] != 'left_only'].drop(columns=['_merge'])

ig_name_matched = ig_unmatched.merge(socialmedia_accounts[['Channel ID', 'Channel Name', 'ServiceName']], 
                                    on='Channel Name', how='inner')

ig_country_df = pd.concat([ig_id_matched, ig_name_matched])
ig_country_df['Channel ID'] = ig_country_df['Channel ID'].fillna(ig_country_df['Channel ID_x'])\
                                                               .fillna(ig_country_df['Channel ID_y'])

ig_country_df['ServiceName'] = ig_country_df['ServiceName'].fillna(ig_country_df['ServiceName_x'])\
                                                       .fillna(ig_country_df['ServiceName_y'])
ig_country_df = ig_country_df.drop(columns=['ServiceName_x', 'ServiceName_y', 
                                            'Channel ID_x', 'Channel ID_y'])


# In[ ]:


ig_country_df.columns


# In[ ]:


ig_country_df.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv", 
                           index=None)


# In[ ]:


'''
# TODO integrate with a tester that makes sure the output file won't change

# country
print("... cleaning country ... ")
ig_country_df = ig_country_df.rename(columns={'ig_metric_breakdown': 'YT-_FBE_codes'})\
                             .merge(country_codes[['YT-_FBE_codes', 'PlaceID']], on='YT-_FBE_codes',
                                    how='left', indicator=True)
print(ig_country_df._merge.value_counts())
ig_country_df = ig_country_df.drop(columns='_merge')
# col selection
print("... selecting columns ... ")
cols = ['w/c', 'Channel ID', 'Channel Name', 'ServiceID', 'PlaceID', 'country_%', 'YT-_FBE_codes', 'week_ending']

ig_country_df = ig_country_df[cols]
ig_country_df.to_excel(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.xlsx", 
                           index=None)
'''


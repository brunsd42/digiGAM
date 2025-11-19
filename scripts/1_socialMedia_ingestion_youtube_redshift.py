#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[1]:


platformID = 'YT-'


# In[2]:


from IPython.display import display

import os
import zipfile

from tqdm import tqdm 
from datetime import datetime

import pandas as pd
pd.set_option('display.max_colwidth', None)

import numpy as np

import re

#import yxdb

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns 

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

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.sample()


# # automated 

# In[30]:


sql_query = f""" 
    SELECT 
        yt_channel_id, 
        yt_country_code,
        yt_metric_id, 
        yt_metric_period, 
        yt_metric_end_time, 
        yt_metric_value 
    FROM 
        redshiftdb.central_insights.yt_channel_insights 
    WHERE
        yt_metric_id = 'views' 
        AND
        yt_channel_id IN ({formatted_channel_ids})
        AND 
        yt_metric_end_time BETWEEN '{gam_info['weekEnding_start']}' AND '{gam_info['weekEnding_end']}'
    ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_geography_redshift_extract.csv"

df = execute_sql_query(sql_query)    
df.to_csv(file, index=False, na_rep='')

yt_views = pd.read_csv(file, keep_default_na=False)

renaming = {'yt_channel_id': 'Channel ID',
            'yt_country_code': 'YT-_FBE_codes',
            'yt_metric_end_time': 'week_ending',
            }
yt_views.rename(columns = renaming, inplace=True)
yt_views['week_ending'] = pd.to_datetime(yt_views['week_ending'])
################################### Testing ################################### 
test_step = 'testing yt_channel_insights return from redshift'

column_name = 'Channel ID'
test_functions.test_filter_elements_returned(yt_views, channel_ids, column_name, "1_YT_6")

test_functions.test_weeks_presence('week_ending', yt_views, week_tester, '1_YT_7', test_step)

test_functions.test_weeks_presence_per_account('week_ending', column_name, yt_views, week_tester, '1_YT_8', test_step)

################################### Testing ################################### 


# In[8]:


# add PlaceID
cols = ['PlaceID', 'YT-_FBE_codes']
yt_views_cleanCountry = yt_views.merge(country_codes[cols], on=['YT-_FBE_codes'], how='left', indicator=True)

################################### Testing ################################### 
test_step = 'adding country codes GAM'

test_functions.test_inner_join(yt_views, country_codes[cols], ['YT-_FBE_codes'], '1_YT_9', test_step, focus='left')

################################### Testing ################################### 
# TODO add HM to GAM lookup
yt_views_cleanCountry[yt_views_cleanCountry._merge == 'left_only']['YT-_FBE_codes'].unique()


# In[9]:


grouped_df_perCountry = yt_views_cleanCountry.groupby([
        'Channel ID',
        'PlaceID',#country
        'week_ending'
    ]).agg({'yt_metric_value': 'sum'}).reset_index()
grouped_df_perCountry = grouped_df_perCountry.rename(columns={'yt_metric_value': 'view_country'})
display(grouped_df_perCountry.sample())

# Group by the specified columns and sum the yt_metric_value
grouped_df_allCountries = yt_views_cleanCountry.groupby([
    'Channel ID',
    'week_ending'
]).agg({'yt_metric_value': 'sum'}).reset_index()
grouped_df_allCountries = grouped_df_allCountries.rename(columns={'yt_metric_value': 'total_view_country'})
#display(grouped_df_allCountries.sample())

country_proportion = grouped_df_allCountries.merge(grouped_df_perCountry, 
                                                   on=['Channel ID', 'week_ending'], 
                                                   how='inner')
country_proportion['country_%'] = (country_proportion['view_country'] / country_proportion['total_view_country'])

################################### Testing ################################### 
# todo: add a test that sums country % and needs to come to a very very very exact 100% (at least 8 decimals)
test_step = 'calculating % country'
cols= ['Channel ID', 'week_ending']
test_functions.test_inner_join(grouped_df_allCountries, grouped_df_perCountry, cols, "1_YT_9", test_step)

test_functions.test_merge_row_count(country_proportion, grouped_df_perCountry, '1_YT_10', test_step)

test_functions.test_percentage(country_proportion,  cols, '1_YT_11', test_step)

test_functions.test_larger_val(country_proportion,  'country_%', '1_YT_12', test_step, val=1)

################################### Testing ################################### 



# In[10]:


automated_country = country_proportion
automated_country = automated_country.merge(week_tester[['w/c', 'week_ending']], 
                                        on=['week_ending'], 
                                        how='left')
country_cols = ['w/c', 'Channel ID', 'PlaceID', 'country_%', ]
automated_country = automated_country[country_cols]
################################### Testing ################################### 
test_step = 'combining country metric'

test_functions.test_merge_row_count(country_proportion, automated_country, '1_YT_17', test_step)
################################### Testing ################################### 




# # manual

# ## import media action

# In[11]:


country_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[['PlaceID', 'YT-_FBE_codes']]


# In[12]:


#TODO: review with minnie for individual exports
#because it contains geography or should we use table instead?

path = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_manual/"
dataframes = []

for filename in os.listdir(path):
    if filename.endswith('.xlsx'):  # Assuming the files are excel files
        
        try:
            file_path = os.path.join(path, filename)
            df = pd.read_excel(file_path, sheet_name='Chart data')
            df['Channel ID'] = filename.split('.')[0].split(' - ')[0]
            df['Channel title'] = filename.split('.')[0].split(' - ')[0]
            df['source_path'] = path+filename
            
            dataframes.append(df)
        except:
            print(filename)
media_action_df = pd.concat(dataframes)

def get_week_dates(date):
    if date.weekday() != 6:  # Check if the date is not a Sunday
        raise ValueError("The input date must be a Sunday.")
    
    from_date = date + pd.Timedelta(days=1)  # Monday after the given Sunday
    to_date = from_date + pd.Timedelta(days=6)  # Sunday after the Monday
    return from_date, to_date

media_action_df['Date'] = pd.to_datetime(media_action_df['Date'])

# Apply the function to get FromDate and ToDate
media_action_df['w/c'], media_action_df['week_ending'] = zip(*media_action_df['Date'].apply(get_week_dates))
media_action_df = media_action_df.rename(columns={'Geography': 'YT-_FBE_codes'})
media_action_df = media_action_df.merge(country_map, on='YT-_FBE_codes', how='left')
# Group by Geography, FromDate, ToDate, and filename to sum Views
media_action_df = media_action_df.groupby(['w/c', 'week_ending', 'Channel ID', 'Channel title', 'PlaceID', 'source_path']).agg({'Views': 'sum'}).reset_index()

media_action_df['Channel Group'] = 'BBC Media Action'

channel_ids = {'Aksi Kita Indonesia': 'aksikitaindo', }
media_action_df['Channel ID'] = media_action_df['Channel ID'].replace(channel_ids)


# In[13]:


_ma_global = media_action_df.groupby(['w/c', 'Channel ID'])['Views'].sum().reset_index()
ma_country_df = media_action_df.merge(_ma_global, on=['w/c', 'Channel ID'], how='left',
                                      suffixes=['_country', '_global'])
ma_country_df['country_%'] = ma_country_df['Views_country'] / ma_country_df['Views_global']
ma_country_df = ma_country_df[automated_country.columns]


# In[14]:


youtube_country = pd.concat([automated_country, ma_country_df])


# ## import Serbian & Sinhala Dataset

# In[15]:


ser_sin_df = pd.read_excel(f"../data/raw/{platformID}/serbian sinhala youtube.xlsx", 
                           sheet_name='SERSIN')
ser_sin_df.rename(columns={'Geography': 'YT-_FBE_codes',
                           'Channel': 'Channel ID',
                           'Total': 'country_%'}, inplace=True)
# join country codes 
ser_sin_df = ser_sin_df.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], on=['YT-_FBE_codes'], indicator=True, how='left')
ser_sin_df = ser_sin_df[country_cols]

ser_sin_df.columns


# In[16]:


# Find rows in additional_df that are NOT in master_df
additional_rows = ser_sin_df[~ser_sin_df.apply(tuple, axis=1).isin(youtube_country.apply(tuple, axis=1))]

# Append new rows to master_df
youtube_country_2 = pd.concat([youtube_country, additional_rows], ignore_index=True)


# # store dataset

# In[17]:


youtube_country_2.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_country_new.csv", 
                         index=None, na_rep='')


# In[ ]:





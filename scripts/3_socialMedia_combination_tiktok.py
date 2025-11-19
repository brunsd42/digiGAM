#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display 

import base64
import json
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

import pandas as pd 
pd.set_option('display.float_format', '{:.00f}'.format)

import os 
import numpy as np
import ast
from sympy import symbols, solve, lambdify

import missingno as msno
import matplotlib.pyplot as plt


# In[2]:


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

from security_config import emplifi_key
from functions import execute_sql_query
import test_functions


# In[3]:


platformID = 'TTK'

# country
cols = ['PlaceID',	'TikTok Codes', gam_info['population_column']]
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='CountryID', usecols=cols, keep_default_na=False )

# service
cols = ['ServiceID', 'Lumen']
service_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='ServiceID', usecols=cols, keep_default_na=False )

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                            sheet_name='GAM Period', keep_default_na=False)

week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new', keep_default_na=False)

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.sample()


# # read in 

# In[4]:


dataframes = []
storage_dir = f"../data/raw/{platformID}/post_level/"
csv_files = [f for f in os.listdir(storage_dir) if f.endswith(".csv")]
for f in csv_files:
    file_path = os.path.join(storage_dir, f)
    try:
        parts = f.replace(".csv", "").split("_")
        file_timeinfo = parts[0]
        platformID = parts[1]
        profile_id = parts[2]
        week_str = parts[3]

        df = pd.read_csv(file_path)
        df["platformID"] = platformID
        df["profile_id"] = profile_id
        df["w/c"] = week_str
        
        if not df.empty:
            dataframes.append(df)
    except pd.errors.EmptyDataError:
        print(f"❌ Could not read file (empty or malformed): {f}")

# Combine all non-empty DataFrames
if dataframes:
    post_level_df = pd.concat(dataframes, ignore_index=True)
    print("✅ Combined DataFrame created.")
    display(post_level_df.head())
else:
    print("🚫 No valid data found to combine.")


# In[5]:


post_level_df = post_level_df.rename(columns={'platformID': 'PlatformID'})
# Count unique weeks per profile
week_counts = post_level_df.groupby('profile_id')['w/c'].nunique().reset_index()
week_counts.columns = ['profile_id', 'number_of_weeks']

# Optional: sort by number of weeks
week_counts = week_counts.sort_values(by='number_of_weeks', ascending=False)

# Display 
print(week_counts)

# Assuming post_level_df is already loaded and contains 'profile_id' and 'w/c'
# Create a pivot table: profiles as rows, weeks as columns
pivot_df = post_level_df.pivot_table(columns='profile_id', index='w/c', aggfunc='size')

# Convert to boolean: True = data exists, False = missing
#pivot_df = pivot_df.astype(bool)

# Visualize missing data
#msno.matrix(pivot_df)
#plt.title("Missing Weeks per TikTok Profile")
#plt.show()


# In[6]:


def extract_author_info(row):
    if pd.isna(row):
        return pd.Series({'id': None, 'name': None, 'url': None})

    if isinstance(row, str):
        try:
            author_dict = ast.literal_eval(row)
        except (ValueError, SyntaxError):
            return pd.Series({'id': None, 'name': None, 'url': None})
    elif isinstance(row, dict):
        author_dict = row
    else:
        return pd.Series({'id': None, 'name': None, 'url': None})

    return pd.Series({
        'id': author_dict.get('id'),
        'name': author_dict.get('name'),
        'url': author_dict.get('url')
    })

# Apply the function
post_level_df[["Channel ID", "Channel Name", "Channel URL"]] = post_level_df['author'].apply(extract_author_info)


# # Views

# In[7]:


minnie_cols_used = {'Date': 'w/c', #minnie has a day by day breakdown and then calculates the average
               'Profile ID': "Channel ID", # author['name'],
               'Profile name': "Channel Name", # author['url'],
               'Profile URL': "Channel URL", #author['id'],
               #'Post detail URL': 'link',
               #'Content ID': 'link', # splice out from link
               'Platform': 'PlatformID', 
               'Content type': 'content_type',
               'Media type': 'media',
               #'Title': '', # missing
               #'Description': '', # missing
               'Content': 'message',
               #'Link URL': '', #unclear
               'View on platform': 'link',
               'Engagements': 'insights_engagements',
               'Total reach': 'insights_reach', #but number is different? 
               'Video length (sec)': 'duration',
               'Video view count': 'insights_video_views',
               'Total video view time (sec)': 'insights_view_time',
               'Average time watched (sec)': 'insights_avg_time_watched',
               'Completion rate': 'insights_completion_rate',
              }

views_df = post_level_df[minnie_cols_used.values()]
views_df['link'] = views_df['link'].fillna('').astype(str)
views_df['content_id'] = views_df['link'].str.split('/').str[-1].str.split('?').str[0]
views_df.head()


# In[8]:


# optional: test video length is all in seconds
print(f"number of entries: {views_df.shape}")
views_df = views_df[~views_df['insights_reach'].isna()]
print(f"number of entries that have reach: {views_df.shape}")

cols_fill_nan = ['insights_avg_time_watched', 'duration', 'insights_reach',
                 'insights_completion_rate']
views_df[cols_fill_nan] = views_df[cols_fill_nan].fillna(0)  # or any other value you'd like


# In[9]:


# Define x and y values for each row
views_df['x1'] = 0
views_df['x2'] = views_df['insights_avg_time_watched']
views_df['x3'] = views_df['duration']
views_df['x output'] = 10

views_df['y1'] = views_df['insights_reach']
views_df['y2'] = views_df['insights_reach'] / 2
views_df['y3'] = views_df['insights_reach'] * views_df['insights_completion_rate']

x_sym, a_sym, b_sym, c_sym = symbols('x a b c')

def find_quadratic_coefficients(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    if len({x1, x2, x3}) < 3:
        return None

    y_expr = a_sym * x_sym**2 + b_sym * x_sym + c_sym

    eq1 = y_expr.subs(x_sym, x1) - y1
    eq2 = y_expr.subs(x_sym, x2) - y2
    eq3 = y_expr.subs(x_sym, x3) - y3

    solutions = solve((eq1, eq2, eq3), (a_sym, b_sym, c_sym))
    return solutions


def apply_quadratic(row):
    if row['x3'] == 0:
        return 100
    if len(set([row['x1'], row['x2'], row['x3']])) < 3:
    # Handle the case where any two x-values are the same
        return 100

    point1 = (row['x1'], row['y1'])
    point2 = (row['x2'], row['y2'])
    point3 = (row['x3'], row['y3'])
    x_val = row['x output']

    coeffs = find_quadratic_coefficients(point1, point2, point3)
    if coeffs is None:
        return np.nan

    # If all y-values are zero, use only the b coefficient
    if row['y1'] == 0 and row['y2'] == 0 and row['y3'] == 0:
        return 10 * coeffs[b_sym]

    # Otherwise, evaluate full quadratic
    y_expr = a_sym * x_sym**2 + b_sym * x_sym + c_sym
    y_val = y_expr.subs({a_sym: coeffs[a_sym], b_sym: coeffs[b_sym], c_sym: coeffs[c_sym], x_sym: x_val})
    return y_val


# Create new column with interpolated values
views_df['30sec_video_views'] = views_df.apply(apply_quadratic, axis=1).astype(float)
views_df['completed_video_views'] = views_df['insights_completion_rate'] * views_df['insights_reach']

conditions = [
    views_df['insights_reach'] == 0,
    (views_df['completed_video_views'].round(0) > views_df['30sec_video_views'].round(0)),
    views_df['insights_avg_time_watched'] <= 10,
    (views_df['insights_avg_time_watched'] > 10) & (views_df['insights_avg_time_watched'] <= 15),
    views_df['insights_reach'] < views_df['30sec_video_views'],
    views_df['duration'] == 0
]

choices = [
    views_df['insights_engagements'],
    views_df['completed_video_views'],
    views_df['insights_engagements'],
    views_df['insights_reach'] * 0.67,
    views_df['insights_reach'] * 0.799,
    views_df['insights_engagements']
]

views_df['final_video_views'] = np.select(conditions, choices, 
                                            default=views_df['30sec_video_views'])


# In[10]:


views_df = views_df.merge(socialmedia_accounts[["Channel ID", "ServiceID", "Linked FB Account"]], 
                              on='Channel ID', how='left', indicator=True)
print(views_df._merge.value_counts())
views_df = views_df.drop(columns=['_merge'])

temp =views_df[views_df['content_id'] == '7353576714619931936']
(temp['completed_video_views'].round(0) > temp['30sec_video_views'].round(0)),

completed = temp['completed_video_views'].round(0).iloc[0]
thirty_sec = temp['30sec_video_views'].round(0).iloc[0]

comparison = completed > thirty_sec
print(comparison)  # Should be False


# In[11]:


cols = ['content_id', 'ServiceID', 'Channel ID', 'Channel Name', 'w/c', 
        'link',
        'final_video_views', 'Linked FB Account'
       ]
views_df = views_df[cols]
views_df.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_views.csv",
                       index=None)


# In[12]:


# YT views per viewer is missing for media action 
yt_views_per_viewer = pd.read_excel("../helper/YT views per viewer_TTKhelper.xlsx")[['w/c', 'Service', 'Value']]
yt_views_per_viewer = yt_views_per_viewer.rename(columns={'Service': 'Lumen', 'Value': 'views_per_viewer'})
yt_views_per_viewer = yt_views_per_viewer.merge(service_codes, on='Lumen', how='left').drop(columns='Lumen')
yt_views_per_viewer.sample()


# In[13]:


views_df['w/c'] = pd.to_datetime(views_df['w/c'])
yt_views_per_viewer['w/c'] = pd.to_datetime(yt_views_per_viewer['w/c'])

views_df_yt = views_df.merge(yt_views_per_viewer, on=['ServiceID', 'w/c'], how='left', indicator=True)

matched = views_df_yt[views_df_yt['_merge'] == 'both'].drop(columns='_merge')
unmatched = views_df_yt[views_df_yt['_merge'] == 'left_only'].drop(columns=['_merge', 'views_per_viewer'])

views_per_viewer_by_service = yt_views_per_viewer.groupby(['ServiceID'])['views_per_viewer'].mean().reset_index()

matched_sec = unmatched.merge(views_per_viewer_by_service, on='ServiceID', how='left', indicator=True)
matched_sec = matched_sec[matched_sec['_merge'] == 'both'].drop(columns='_merge')

views_scaled = pd.concat([matched, matched_sec])
views_scaled['engaged_users'] = views_scaled['final_video_views']/(views_scaled['views_per_viewer']*1.14)
views_scaled.columns


# # Country 

# In[14]:


country_df = post_level_df.copy()


# In[15]:


# Step 1: Parse the stringified list of country-percentage dictionaries
country_df['parsed_viewers'] = country_df['insights_viewers_by_country'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)

# Step 2: Explode the parsed list into separate rows
exploded_df = country_df.explode('parsed_viewers').reset_index(drop=True)

# Step 3: Extract 'country' and 'percentage' from each dictionary
exploded_df[['country', 'percentage']] = exploded_df['parsed_viewers'].apply(
    lambda entry: pd.Series({
        'country': entry.get('country') if isinstance(entry, dict) else None,
        'percentage': entry.get('percentage') if isinstance(entry, dict) else None
    })
)

# Step 4: Drop the intermediate column
exploded_df.drop(columns=['parsed_viewers'], inplace=True)
exploded_df['country'] = exploded_df['country'].replace('Others', 'ZZ')
exploded_df['country'] = exploded_df['country'].fillna('ZZ')


# In[20]:


exploded_df = exploded_df.rename(columns={'country': 'TikTok Codes'})
ttk_country_all = exploded_df.merge(country_codes[['TikTok Codes', 'PlaceID']], on='TikTok Codes', how='left',
                 indicator=True)

print(f"mismatches? \n{ttk_country_all._merge.value_counts()}")
ttk_country_all = ttk_country_all.drop(columns='_merge')

# Remove unknown countries (UNK)
ttk_country = ttk_country_all[ttk_country_all['PlaceID'] != 'UNK'].copy()

# Compute sum of percentages per video
sum_per_video = ttk_country.groupby('id')['percentage'].transform('sum')

# Rescale percentages
ttk_country['rescaled_percentage'] = ttk_country['percentage'] / sum_per_video

# Optional: Check that each video sums to 1
check = ttk_country.groupby('id')['rescaled_percentage'].sum()
check[~np.isclose(check, 1, atol=1e-6)]

country_cols = ['Channel ID', 'link', 'PlaceID', 'rescaled_percentage', 'w/c', 'PlatformID']
ttk_country= ttk_country[country_cols]
ttk_country.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv",
                  index=None, na_rep='')


# # combine views & country
# 
# country is a percentage by video. In the next section the actual metric is calculated (video views) per country. This is then summed to the profile level. 
# As the average is larger than 1 the country view is rebased 
# 

# In[21]:


# post_url: link
ttk_country['w/c'] = pd.to_datetime(ttk_country['w/c'])

ttk_country = ttk_country.drop(columns=['_merge'], errors='ignore')
views_scaled = views_scaled.drop(columns=['_merge'], errors='ignore')

ttk_data = ttk_country.merge(views_scaled, on=['Channel ID', 'link', 'w/c'], how='outer',
                                indicator=True)
print(f"mismatches between datasets! unreviewed yet\n {ttk_data._merge.value_counts()}")
ttk_data = ttk_data[ttk_data['_merge'] == 'both'].drop(columns=['_merge'])

ttk_data['country_views_video_level'] = ttk_data["final_video_views"] * ttk_data["rescaled_percentage"]
ttk_perCountry = ttk_data.groupby(['w/c', 'Channel ID', 'ServiceID',
                                'PlaceID'])['country_views_video_level'].sum().rename('country_views_profile_level').reset_index()
ttk_global = ttk_data.groupby(['w/c', 'ServiceID','Channel ID', 
                              ])['country_views_video_level'].sum().rename('global_views_profile_level').reset_index()

ttk_df = ttk_perCountry.merge(ttk_global, on=['ServiceID', 'w/c', 'Channel ID'], how='outer', 
                           indicator=True)
print(f"no mismatches \n {ttk_df._merge.value_counts()}")
ttk_df = ttk_df.drop(columns=['_merge'])

ttk_df['profile_country_views_%'] = ttk_df['country_views_profile_level']/ttk_df['global_views_profile_level']

ttk_df = ttk_df.merge(views_scaled, on=['ServiceID', 'Channel ID', 'w/c'], how='inner')
ttk_df['uv_by_country'] =  ttk_df['engaged_users'] * ttk_df['profile_country_views_%']

cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', ]
ttk_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv", 
                     index=None)


# In[ ]:





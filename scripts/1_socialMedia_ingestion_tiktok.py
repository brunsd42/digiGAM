#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

from security_config import emplifi_key
from functions import execute_sql_query
import test_functions


# In[3]:


platformID = 'TTK'

# country
cols = ['PlaceID',	'TikTok Codes']
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='CountryID', usecols=cols, keep_default_na=False )

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


# # ingest data

# In[4]:


post_url = "https://api.emplifi.io/3/tiktok/profile/posts"
               
# create secret token for API authentication
secret_token = f"{emplifi_key['access_token']}:{emplifi_key['secret']}"
encoded_secret_token = base64.b64encode(secret_token.encode('utf-8')).decode('utf-8')

# authentication using secret token
headers = {
    "Authorization": f"Basic {encoded_secret_token}"
}


# In[5]:


# function to get insights (post level) from user profile
def get_post_level_insights(start_date, end_date, profile_id):

    total_posts = [] # create empty list to contain the posts
    after_param = None # after parameter for going to the next page (Pagination)

    # API parameters to get posts from user profile
    payload = {
        "profiles": [profile_id],
        "date_start": start_date,
        "date_end": end_date,
        "fields": [
            "attachments","author","authorId","content_type","created_time","duration","id",
            "link","media","message","post_labels","insights_avg_time_watched","insights_comments",
            "insights_completion_rate","insights_engagements","insights_impressions",
            "insights_impressions_by_traffic_source","insights_likes","insights_reach",
            "insights_reach_engagement_rate","insights_shares","insights_video_views","insights_view_time",
            "insights_viewers_by_country"
        ],
        "sort": [{"field": "created_time", "order": "desc"}],
        "limit": 100,
    }

    # get posts from profile using api parameters
    response = requests.post(post_url, headers=headers, json=payload)
    
    # Check if the response was successful
    if response.status_code != 200:
        print(f"❌ API request failed with status code {response.status_code} for profile {profile_id}, {start_date}")
        print(response.text)
        return pd.DataFrame()
    
    try: # check if response can be turned to json format
        data = response.json()
    except json.JSONDecodeError:
        print("Invalid JSON content returned by API")
        exit()

    # get list of posts from response
    posts = data.get("data", {}).get("posts", [])

    # add posts to total posts list
    total_posts.extend(posts)

    # get after parameter for pagination
    after_param = data.get("data", {}).get("next", None)

    # start loop to get remaining pages
    while True: # REQUIREMENT 3: Loop the request to get all published posts within the time period
        # stop loop if there is no 'next' value (i.e. no next page)
        if not after_param:
            break

        # parameter to get next page's posts
        payload = {
            "after": after_param
        }

        # get posts
        response = requests.post(post_url, headers=headers, json=payload)
        try:
            data = response.json()
        except json.JSONDecodeError:
            print("Invalid JSON content returned by API")
            exit()

        # extract list of posts from response
        posts = data.get("data", {}).get("posts", [])

        # stop loop if there are no more posts
        if not posts:
            break

        # add new set of posts to total posts
        total_posts.extend(posts)

        # get after parameter for pagination
        after_param = data.get("data", {}).get("next", None)

    # store extracted posts into a dataframe
    df = pd.DataFrame(total_posts)
    if len(df) == 0:
        print(f"empty dataset! response status text: {response.text}")
    return df


# In[6]:


# Directory to store weekly data
storage_dir = f"../data/raw/{platformID}/post_level/"
os.makedirs(storage_dir, exist_ok=True)

for profile_id in tqdm(socialmedia_accounts['Channel ID'].unique()):
    # Sort weeks from newest to oldest
    for week in week_tester['w/c'].sort_values(ascending=False):
        
        if week > datetime.now():
            break
        end_date = week + pd.DateOffset(days=(6 - week.weekday()))
        week_str = week.strftime("%Y-%m-%d")
        filename = f"{storage_dir}/{gam_info['file_timeinfo']}_{platformID}_{profile_id}_{week_str}.csv"

        if os.path.exists(filename):
            continue
        else:
            print(f"🔄 Fetching data for {profile_id} on week {week_str}...")
            df = get_post_level_insights(week_str, end_date.strftime("%Y-%m-%d"), profile_id)
            
            if df.empty:
                print(f"⚠️ No data returned for {profile_id} on week {week_str}. Skipping save.")
                continue
            else:
                df["platformID"] = platformID
                df["profile_id"] = profile_id
                df["w/c"] = week
        
            df.to_csv(filename, index=False)
            print(f"✅ Saved to {filename}")


# In[7]:


country_df = post_level_df.copy()


# In[ ]:


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
exploded_df.head()


# In[ ]:


exploded_df = exploded_df.rename(columns={'country': 'TikTok Codes'})
ttk_country = exploded_df.merge(country_codes[['TikTok Codes', 'PlaceID']], on='TikTok Codes', how='left',
                 indicator=True)

print(f"mismatches? \n{ttk_country._merge.value_counts()}")
ttk_country = ttk_country.drop(columns='_merge')


# In[ ]:


country_cols = ['Channel ID', 'link', 'PlaceID', 'percentage', 'w/c', 'PlatformID']
ttk_country.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv",
                  index=None, keep_default_na=False)


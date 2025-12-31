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

import time


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
from functions import lookup_loader
import test_functions


# In[3]:


platformID = 'TTK'
lookup = lookup_loader(gam_info, platformID, '1',)
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']


'''# country
country_cols = ['PlaceID',	'TikTok Codes']
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='CountryID', usecols=country_cols, keep_default_na=False )

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
                                            & (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})


### RUN TESTS
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_1_0", f"{platformID}_1_1", f"{platformID}_1_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_1_3", f"{platformID}_1_4", f"{platformID}_1_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_1_6", f"{platformID}_1_7", f"{platformID}_1_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")
'''


# # ingest data

# In[4]:


post_url = "https://api.emplifi.io/3/tiktok/profile/posts"
               
# create secret token for API authentication
secret_token = f"{emplifi_key['access_token']}:{emplifi_key['secret']}"
encoded_secret_token = base64.b64encode(secret_token.encode('utf-8')).decode('utf-8')

# authentication using secret token
headers_bau = {
    "Authorization": f"Basic {encoded_secret_token}"
}



# In[5]:


# function to get insights (post level) from user profile
def get_post_level_insights(start_date, end_date, profile_id, headers):

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

MAX_CALLS = 500
PERIOD = 3600  # seconds (1 hour)

start_time = time.time()
request_count = 0

# Sort weeks from newest to oldest
for week in week_tester['w/c'].sort_values(ascending=False):
    print(f"processing {week}")
    for profile_id in tqdm(socialmedia_accounts['Channel ID'].tolist()):
        # Check if we hit the limit
        if request_count >= MAX_CALLS:
            elapsed = time.time() - start_time
            if elapsed < PERIOD:
                wait = PERIOD - elapsed
                print(f"⏳ Hit {MAX_CALLS} requests. Waiting {wait:.1f}s until hour resets...")
                time.sleep(wait)
            # Reset for next hour
            start_time = time.time()
            request_count = 0

        if week > datetime.now():
            continue
        end_date = week + pd.DateOffset(days=(6 - week.weekday()))
        week_str = week.strftime("%Y-%m-%d")
        filename = f"{gam_info['file_timeinfo']}_{platformID}_{profile_id}_{week_str}.csv"

        if os.path.exists(filename):
            continue
        else:
            print(f"🔄 Fetching data for {profile_id} on week {week_str}...")
            df = get_post_level_insights(week_str, end_date.strftime("%Y-%m-%d"), 
                                         profile_id, headers_bau)
            cols_that_must_not_be_empty = ['author', 'insights_viewers_by_country',
                                           'insights_avg_time_watched', 'duration', 
                                           'insights_reach', 'insights_completion_rate']    
            if df.empty:
                print(f"⚠️ No data returned for {profile_id} on week {week_str}. Skipping save.")
                continue
            
            elif df[cols_that_must_not_be_empty].isna().all(axis=0).any():
                issues_dir = f"../data/raw/{platformID}/post_level/issues"
                os.makedirs(issues_dir, exist_ok=True)
                df.to_csv(os.path.join(issues_dir, filename), index=False)
                print(f"⚠️ Partial data returned for {profile_id} on week {week_str}. Saved in issue folder: {issues_dir}")
                continue
                
            else:
                df["platformID"] = platformID
                df["profile_id"] = profile_id
                df["w/c"] = week
        
            df.to_csv(f"{storage_dir}/{filename}", index=False)
            print(f"✅ Saved to {filename}")
            


# In[ ]:





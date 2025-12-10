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
# Set pandas option to format floats as percentages with 2 decimal places
pd.set_option('display.float_format', '{:.2%}'.format)

import os 
import shutil
import numpy as np
import ast

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
from functions import execute_sql_query, gnl_expander
import test_functions
import functions 


# In[3]:


platformID = 'TTK'

# country
country_cols = ['PlaceID',	'TikTok Codes', gam_info['population_column']]
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='CountryID', usecols=country_cols, keep_default_na=False )

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                            sheet_name='GAM Period', keep_default_na=False)
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

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

# service
cols = ['ServiceID', 'Lumen']
service_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                              sheet_name='ServiceID', usecols=cols, keep_default_na=False )

### RUN TESTS
test_functions.test_lookup_files(country_codes, ['PlaceID'], [f"{platformID}_3_0", f"{platformID}_3_1", f"{platformID}_3_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3_3", f"{platformID}_3_4", f"{platformID}_3_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_3_6", f"{platformID}_3_7", f"{platformID}_3_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


test_functions.test_lookup_files(service_codes, ['ServiceID', 'Lumen'], [f"{platformID}_3_9", f"{platformID}_3_10", f"{platformID}_3_11"], 
                                 test_step = "lookup files - ensuring service is correct")



# # read in 

# In[4]:


cols_that_must_not_be_empty = ['author', 'insights_viewers_by_country',
                               'insights_avg_time_watched', 'duration', 
                               'insights_reach', 'insights_completion_rate']
dataframes = []
storage_dir = f"../data/raw/{platformID}/post_level/"
csv_files = [f for f in os.listdir(storage_dir) if f.endswith(".csv")]
for f in csv_files:
    file_path = os.path.join(storage_dir, f)
    try:
        parts = f.replace(".csv", "").split("_")
        file_timeinfo = parts[0]
        if platformID != parts[1]:
            print('something is wrong with platformID in filename!')
        platformID = parts[1]
        profile_id = parts[2]
        week_str = parts[3]

        df = pd.read_csv(file_path)
        # test columns that must not be nan here -> move them to issues folder
        #Detect any required column that is entirely NaN
        entirely_nan_mask = df[cols_that_must_not_be_empty].isna().all(axis=0)
        entirely_nan_cols = entirely_nan_mask[entirely_nan_mask].index.tolist()
        
        if len(entirely_nan_cols) > 0:
            issues_dir = f"../data/raw/{platformID}/post_level/issues"
            os.makedirs(issues_dir, exist_ok=True)
                
            dest_path = os.path.join(issues_dir, f)
            shutil.move(file_path, dest_path)
            print(f"⚠️ Moved to issues (entirely NaN cols {entirely_nan_cols}): {f}")
            continue
            
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


# In[7]:


post_level_df['w/c'].sort_values().unique()


# # Views

# In[8]:


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


# In[9]:


# optional: test video length is all in seconds
print(f"number of entries: {views_df.shape}")
views_df = views_df[~views_df['insights_reach'].isna()]
print(f"number of entries that have reach: {views_df.shape}")

cols_fill_nan = ['insights_avg_time_watched', 'duration', 'insights_reach',
                 'insights_completion_rate']
views_df[cols_fill_nan] = views_df[cols_fill_nan].fillna(0)  # or any other value you'd like


# In[10]:


# Define x and y values for each row
views_df['x1'] = 0
views_df['x2'] = views_df['insights_avg_time_watched']
views_df['x3'] = views_df['duration']
views_df['x output'] = 10

views_df['y1'] = views_df['insights_reach']
views_df['y2'] = views_df['insights_reach'] / 2
views_df['y3'] = views_df['insights_reach'] * views_df['insights_completion_rate']

def apply_quadratic_fast(row):
    x_vals = np.array([row['x1'], row['x2'], row['x3']], dtype=float)
    y_vals = np.array([row['y1'], row['y2'], row['y3']], dtype=float)

    if len(set(x_vals)) < 3 or row['x3'] == 0:
        return 100

    # Build matrix and solve
    A = np.vstack([x_vals**2, x_vals, np.ones(3)]).T
    coeffs = np.linalg.solve(A, y_vals)

    # Evaluate at x output
    return coeffs[0]*row['x output']**2 + coeffs[1]*row['x output'] + coeffs[2]

# Create new column with interpolated values
views_df['30sec_video_views'] = views_df.apply(apply_quadratic_fast, axis=1).astype(float)
views_df['completed_video_views'] = views_df['insights_completion_rate'] * views_df['insights_reach']

conditions = [
    views_df['insights_reach'] == 0,
    (views_df['completed_video_views'].round(0) > views_df['30sec_video_views'].round(0)),
    views_df['insights_reach'] < views_df['30sec_video_views'],
    views_df['duration'] == 0
]

choices = [
    views_df['insights_engagements'],
    views_df['completed_video_views'],
    views_df['insights_reach'] * 0.799,
    views_df['insights_engagements']
]

views_df['final_video_views'] = np.select(conditions, choices, 
                                            default=views_df['30sec_video_views'])


# In[11]:


'''views_df[
    (views_df['Channel ID'] == 'c02ca653-c3b6-4b34-b210-711e12f9eb2d') &
    (views_df['w/c'] == '2025-08-04') ].sort_values('final_video_views', ascending=False)'''


# In[12]:


views_df_full = views_df.merge(socialmedia_accounts[['Channel ID', 'ServiceID', 'Linked FB Account']],
               on='Channel ID', how='left')
test_functions.test_inner_join(views_df, socialmedia_accounts[['Channel ID', 'ServiceID', 'Linked FB Account']], 
                               ['Channel ID'], 
                               f"12_{platformID}_engagements", 
                               test_step='checking social media accounts in lookup, adding service',
                               focus='left')


# In[13]:


file_path = f"../data/processed/{platformID}"
os.makedirs(file_path, exist_ok=True)

cols = ['content_id', 'ServiceID', 'Channel ID', 'Channel Name', 'w/c', 
        'link',
        'final_video_views', 'Linked FB Account'
       ]
views_df_full = views_df_full[cols]
views_df_full.to_csv(f"{file_path}/{gam_info['file_timeinfo']}_{platformID}_views.csv",
                       index=None)


# In[14]:


# YT views per viewer is missing for media action 
yt_views_per_viewer = pd.read_excel("../data/stale/YT views per viewer_TTKhelper.xlsx")[['w/c', 'Service', 'views_per_viewer']]
yt_views_per_viewer = yt_views_per_viewer.rename(columns={'Service': 'Lumen'})
yt_views_per_viewer = yt_views_per_viewer.merge(service_codes, on='Lumen', how='left').drop(columns='Lumen')
yt_views_per_viewer = gnl_expander(yt_views_per_viewer)
yt_views_per_viewer.sample()


# In[15]:


views_df_full['w/c'] = pd.to_datetime(views_df_full['w/c'])
yt_views_per_viewer['w/c'] = pd.to_datetime(yt_views_per_viewer['w/c'])

views_df_yt = views_df_full.merge(yt_views_per_viewer, on=['ServiceID', 'w/c'], how='left', indicator=True)

matched = views_df_yt[views_df_yt['_merge'] == 'both'].drop(columns='_merge')
unmatched = views_df_yt[views_df_yt['_merge'] == 'left_only'].drop(columns=['_merge', 'views_per_viewer'])

views_per_viewer_by_service = yt_views_per_viewer.groupby(['ServiceID'])['views_per_viewer'].mean().reset_index()

matched_sec = unmatched.merge(views_per_viewer_by_service, on='ServiceID', how='left', indicator=True)
matched_sec = matched_sec[matched_sec['_merge'] == 'both'].drop(columns='_merge')

views_scaled = pd.concat([matched, matched_sec])
views_scaled['engaged_users'] = views_scaled['final_video_views']/(views_scaled['views_per_viewer']*1.14)
views_scaled.columns


# In[16]:


views_scaled['w/c'].sort_values().unique()


# # Country 

# In[17]:


country_df = post_level_df.copy()


# In[18]:


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


# In[19]:


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


# In[20]:


import numpy as np
import pandas as pd

# 0) Weekly mean per channel/place (your step)
ttk_country_avg_channel = (
    ttk_country
    .groupby(['Channel ID', 'w/c', 'PlaceID', 'PlatformID'])['rescaled_percentage']
    .mean()
    .reset_index()
)

# 1) Ensure datetime and sort
ttk_country_avg_channel['w/c'] = pd.to_datetime(ttk_country_avg_channel['w/c'])
ttk_country_avg_channel = ttk_country_avg_channel.sort_values(['Channel ID', 'PlaceID', 'w/c']).copy()

# 2) Compute prev-4-week rolling average per (Channel ID, PlaceID)
ttk_country_avg_channel['prev4wk_avg'] = (
    ttk_country_avg_channel
    .groupby(['Channel ID', 'PlaceID'], sort=False)['rescaled_percentage']
    .transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
)

# 3) For each channel, create the missing "last week" (next calendar week) and impute per PlaceID
imputed_rows = []

# Choose your week anchor to match 'w/c' (e.g., W-MON if w/c is Monday)
week_freq = 'W-MON'  # change to 'W-SUN' etc. if needed

for ch, ch_grp in ttk_country_avg_channel.groupby('Channel ID'):
    ch_grp = ch_grp.sort_values('w/c').copy()
    if ch_grp.empty:
        continue

    # Latest known week for this channel
    last_week = ch_grp['w/c'].max()

    # Compute the next week date aligned to your anchor
    # If your w/c values are already Mondays, next week is last_week + 7 days.
    # Alternatively, use date_range to get the next anchor precisely:
    next_week = (pd.date_range(start=last_week, periods=2, freq=week_freq)[-1])

    # If next_week already exists in data for this channel, skip
    if ((ch_grp['w/c'] == next_week).any()):
        continue

    # Build per-place imputation for the missing week using prev4wk_avg
    places = ch_grp['PlaceID'].unique()
    platform_id = ch_grp['PlatformID'].dropna().iloc[0] if not ch_grp['PlatformID'].dropna().empty else np.nan

    # Collect prev4wk_avg per place from the latest rows in the channel history
    # We take the most recent available prev4wk_avg for each place.
    rows_for_week = []
    for place in places:
        place_grp = ch_grp[ch_grp['PlaceID'] == place].copy()
        # The prev4wk_avg is already computed per row; pick the last non-NaN value
        val = place_grp['prev4wk_avg'].dropna().iloc[-1] if not place_grp['prev4wk_avg'].dropna().empty else np.nan
        rows_for_week.append({'Channel ID': ch, 'PlaceID': place, 'PlatformID': platform_id,
                              'w/c': next_week, 'imputed_share': val})

    imputed_df = pd.DataFrame(rows_for_week)

    # If all places are NaN (no history at all), assign a uniform split
    if imputed_df['imputed_share'].isna().all():
        imputed_df['imputed_share'] = 1.0 / len(imputed_df)
    else:
        # Normalize so sums to 1 across places for that channel-week
        total = imputed_df['imputed_share'].sum(skipna=True)
        if pd.isna(total) or total == 0:
            # If total is 0 or NaN after partial NaNs, use uniform
            imputed_df['imputed_share'] = 1.0 / len(imputed_df)
        else:
            # For any NaNs, set to 0 before normalization (optional: or fill with channel-place median)
            imputed_df['imputed_share'] = imputed_df['imputed_share'].fillna(0)
            imputed_df['imputed_share'] = imputed_df['imputed_share'] / imputed_df['imputed_share'].sum()

    # Finalize column names to match your dataset
    imputed_df = imputed_df.rename(columns={'imputed_share': 'rescaled_percentage'})
    imputed_rows.append(imputed_df)

# 4) Append imputed "last week" rows (if any) to the weekly dataset
if imputed_rows:
    imputed_last_weeks = pd.concat(imputed_rows, ignore_index=True)
else:
    imputed_last_weeks = pd.DataFrame(columns=['Channel ID', 'PlaceID', 'PlatformID', 'w/c', 'rescaled_percentage'])

# Optional: flag these rows
imputed_last_weeks['is_imputed'] = True

# 5) Combine: keep original observed weeks + the newly imputed last weeks
weekly_with_imputed_last = pd.concat([ttk_country_avg_channel[['Channel ID','PlaceID','PlatformID','w/c','rescaled_percentage']].assign(is_imputed=False),
                                      imputed_last_weeks],
                                     ignore_index=True)

# 6) (Optional) save
# weekly_with_imputed_last.to_csv(
#     f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_weekly_with_last_week_imputed.csv",
#     index=False
# )
weekly_with_imputed_last[weekly_with_imputed_last['is_imputed'] == True]


# # combine views & country
# 
# country is a percentage by video. In the next section the actual metric is calculated (video views) per country. This is then summed to the profile level. 
# As the average is larger than 1 the country view is rebased 
# 

# In[21]:


ttk_country.columns


# In[22]:


# 1. Convert week column to datetime
ttk_country['w/c'] = pd.to_datetime(ttk_country['w/c'])

# 2. Remove merge indicators if present
ttk_country = ttk_country.drop(columns=['_merge'], errors='ignore')
views_scaled = views_scaled.drop(columns=['_merge'], errors='ignore')

# 3. Initial merge: country-level data with scaled views
merged_initial = ttk_country.merge(
    views_scaled,
    on=['Channel ID', 'link', 'w/c'],
    how='outer',
    indicator=True
)
print(f"Initial merge mismatches:\n{merged_initial._merge.value_counts()}")

# deal with country
unmatched_country = merged_initial[merged_initial['_merge'] == 'left_only']
# for each channel / w/c combination I want the country average of the previous 4 weeks to be the fill in for the missing data
merged_country_avg = weekly_with_imputed_last[['w/c', 'Channel ID', 'PlaceID', 'rescaled_percentage']].merge(unmatched_country[views_scaled.columns],
    on=['Channel ID', 'w/c'],
    how='outer',
    indicator=True
)
print(f"Country second merge mismatches:\n{merged_country_avg._merge.value_counts()}")

# 4. Identify unmatched rows (right_only - country only)
unmatched_views = merged_initial[merged_initial['_merge'] == 'right_only']

# 5. Compute weekly averages for unmatched rows
weekly_avg = ttk_country.groupby(['Channel ID', 'w/c'])['rescaled_percentage'].mean().reset_index()
merged_weekly_avg = weekly_avg.merge(
    unmatched_views[views_scaled.columns],
    on=['Channel ID', 'w/c'],
    how='outer',
    indicator=True
)
print(f"Mismatches after weekly average merge:\n{merged_weekly_avg._merge.value_counts()}")

# 6. Identify remaining unmatched rows and compute channel-level averages
still_unmatched = merged_weekly_avg[merged_weekly_avg['_merge'] == 'right_only']
channel_avg = ttk_country.groupby(['Channel ID'])['rescaled_percentage'].mean().reset_index()
merged_channel_avg = channel_avg.merge(
    still_unmatched[views_scaled.columns],
    on=['Channel ID'],
    how='outer',
    indicator=True
)
print(f"Mismatches after channel-level average merge:\n{merged_channel_avg._merge.value_counts()}")

# 7. Combine all data sources (same as original logic)
combined_data = pd.concat(
    [merged_initial, merged_country_avg, merged_weekly_avg, merged_channel_avg],
    ignore_index=True
).drop(columns=['_merge'])

# 8. Calculate country-level views at video level
combined_data['country_views_video_level'] = (
    combined_data['final_video_views'] * combined_data['rescaled_percentage']
)

# 9. add population column 
combined_data = combined_data.merge(country_codes[['PlaceID', gam_info['population_column']]],
                                    on=['PlaceID'], how='left')


# In[23]:


# 10. Apply Sainsbury formula for country-level views
deduplicated_datasets = []
for channel in tqdm(combined_data['Channel ID'].unique()):
    temp = combined_data[combined_data['Channel ID'] == channel]
    channel_uv_by_country = pd.crosstab(
                                        index = [ temp['PlaceID'], 
                                                  temp['PlatformID'],
                                                  temp['ServiceID'],
                                                  temp['Channel ID'],
                                                  temp[gam_info['population_column']],
                                                  temp['w/c']],
                                        columns = temp['link'],
                                        values =  temp['country_views_video_level'],
                                        aggfunc='sum'
                                    )
    link_ids = channel_uv_by_country.columns
    channel_uv_by_country = channel_uv_by_country.reset_index().fillna(0)
    channel_uv_by_country = functions.sainsbury_formula(channel_uv_by_country, 
                                                        gam_info['population_column'],
                                                        link_ids, 
                                                        'country_views_video_level')
    channel_uv_by_country = channel_uv_by_country.drop(columns=link_ids)
    deduplicated_datasets.append(channel_uv_by_country)

dedupli_df = pd.concat(deduplicated_datasets)
        


# In[24]:


# 11. Aggregate to profile level (country-specific)
country_profile_views = dedupli_df.groupby(
    ['w/c', 'Channel ID', 'ServiceID', 'PlaceID']
)['country_views_video_level'].sum().rename('country_views_profile_level').reset_index()

# 12. Aggregate to global profile level
global_profile_views = dedupli_df.groupby(
    ['w/c', 'ServiceID', 'Channel ID']
)['country_views_video_level'].sum().rename('global_views_profile_level').reset_index()

# 13. Merge country and global profile views
profile_views = country_profile_views.merge(
    global_profile_views,
    on=['ServiceID', 'w/c', 'Channel ID'],
    how='outer',
    indicator=True
)
print(f"Profile-level merge check:\n{profile_views._merge.value_counts()}")
profile_views = profile_views.drop(columns=['_merge'])

# 14. Calculate percentage contribution of country views
profile_views['profile_country_views_%'] = (
    profile_views['country_views_profile_level'] / profile_views['global_views_profile_level']
)

# 15. Merge back with scaled views for user-level metrics
ttk_df = profile_views.merge(
    views_scaled,
    on=['ServiceID', 'Channel ID', 'w/c'],
    how='inner'
)

# 16. Calculate UV by country
ttk_df['uv_by_country'] = (
    ttk_df['engaged_users'] * ttk_df['profile_country_views_%']
)


# In[25]:


print(ttk_df.shape)
ttk_df = ttk_df.dropna(subset='uv_by_country')
print(ttk_df.shape)

cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', ]
ttk_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv", 
                     index=None)


# In[31]:


ttk_df[ttk_df['w/c'] == '2025-12-01']['ServiceID'].unique()


# In[30]:


ttk_df[ttk_df['w/c'] == '2025-11-24']['ServiceID'].unique()


# In[ ]:





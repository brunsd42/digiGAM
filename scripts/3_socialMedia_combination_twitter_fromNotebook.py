#!/usr/bin/env python
# coding: utf-8

# In[1]:


platformID = 'TWI'


# ## Status 
# twitter data is currently ingested and the process of adding facebook factors to it has been started, however the output files still show significant differences to minnie's dataset. 
# next step here is retesting the input files and then step by step through the combination process. 
# 
# twitter business unit and aggregated services is currently calculated by using minnie's dataset (helper/tw_minnie_preBU.csv)

# In[2]:


from datetime import datetime
import pandas as pd
import numpy as np
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

from functions import lookup_loader, gnl_expander
import test_functions


# In[4]:


lookup = lookup_loader(gam_info, platformID, '3',
                       with_country=True, country_col=['TWI_CountryName'])
week_tester = lookup['week_tester']
socialmedia_accounts = lookup['socialmedia_accounts']
country_codes = lookup['country_codes']


'''channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
'''


# # ingestion

# In[5]:


tw_activity_df_raw = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv",
                             keep_default_na=False)
tw_activity_df_raw['w/c'] = pd.to_datetime(tw_activity_df_raw['w/c'])
tw_activity_df = tw_activity_df_raw.merge(week_tester[['w/c', 'WeekNumber_finYear']], 
                                          on='w/c', how='left')
test_functions.test_inner_join(tw_activity_df, week_tester[['w/c', 'WeekNumber_finYear']],
                               ['w/c'],
                               f"{platformID}_3_09",
                               focus='left')

tw_activity_df.sample()


# In[6]:


tw_country_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv",
                            low_memory=False)

tw_country_df['Channel ID'] = tw_country_df['Channel ID'].fillna('')
cols = ['Channel ID', 'Linked FB Account', 'WeekNumber_finYear', 'PlaceID', 'Engagement %']
tw_country_df = tw_country_df[cols]

tw_country_df['Linked FB Account'] = tw_country_df['Linked FB Account'].apply(
    lambda x: str(int(x)) if pd.notnull(x) and str(x).strip() != '' else ''
)

# TODO
# test uniqueness - account / week 
tw_country_df.sample()


# # combine 

# In[7]:


'''tw_activity_country[tw_activity_country['_merge'] == 'left_only'].merge(socialmedia_accounts[['Channel ID', 'ServiceID']], on ='Channel ID', how='left')
'''


# In[8]:


tw_activity_country = tw_activity_df.merge(tw_country_df, on=['Channel ID', 'WeekNumber_finYear'], 
                                           how='left', indicator=True)

test_functions.test_inner_join(tw_activity_df, tw_country_df, ['Channel ID', 'WeekNumber_finYear'], 
                               f"{platformID}_3_10", 
                               test_step='joining activity & country - first', focus='left')
tw_activity_country.sample()


# In[9]:


left_over = tw_activity_country[tw_activity_country._merge == 'left_only'].drop(columns='_merge')
final_1 = tw_activity_country[tw_activity_country._merge == 'both'].drop(columns='_merge')

# groupby 
grouped_df = tw_country_df.groupby(['Channel ID',  'Linked FB Account', 'PlaceID']).agg({
    'Engagement %': 'mean',
}).reset_index()

# twitter_activity_metadata.columns: to keep the columns from the initial merge and loose all those that are empty anyway
final_2 = left_over[tw_activity_df.columns].merge(grouped_df, on='Channel ID', how='left')#.drop(columns='_merge')
test_functions.test_inner_join(left_over, grouped_df, ['Channel ID'], 
                               f"{platformID}_3_11",
                               test_step='joining activity & country - second', 
                                focus='left')


# In[10]:


cols = ['Channel ID', 'tweet_engagements', 'video_video_views',
        'w/c', 'WeekNumber_finYear', 'Linked FB Account',
        'PlaceID', 'Engagement %' ]
final_df_slim = pd.concat([final_1, final_2])[cols]

final_df_slim['tweet_engagements'] = np.where(final_df_slim['tweet_engagements']<0, 0, final_df_slim['tweet_engagements'])

final_df = final_df_slim.merge(socialmedia_accounts[['Channel ID','ServiceID']], on='Channel ID', how='left')
test_functions.test_inner_join(final_df_slim, socialmedia_accounts[['Channel ID','ServiceID']], ['Channel ID'], 
                               f"{platformID}_3_12", test_step='adding ServiceID', focus='left')

# TODO: find out why there are so many duplicates
print(final_df.shape)
print(final_df.drop_duplicates().shape)

# file is used to compare to minnie's dataset:
final_df.to_csv(f"../data/processed/{platformID}/temp_{gam_info['file_timeinfo']}_metric_country.csv", index=None)
final_df.sample()


# In[11]:


# handle if country == 'other'
regular_country = final_df[final_df['PlaceID'] != 'UNK']
regular_country_100 = final_df[final_df['PlaceID'] != 'UNK']
regular_country_100 = regular_country_100.groupby(['Channel ID', 'WeekNumber_finYear'])['Engagement %'].sum().reset_index()

rescaled_df = regular_country.merge(regular_country_100, on=['Channel ID', 'WeekNumber_finYear'], how='left',
                                    suffixes=['_', 'newTotal'])

rescaled_df['engagement_%'] = rescaled_df["Engagement %_"]/rescaled_df["Engagement %newTotal"]


# ## facebook factor 

# In[12]:


fb_factor = pd.read_excel("../data/stale/FB Factor for IG and TW.xlsx").drop_duplicates()
fb_factor['FB Page ID'] = fb_factor['FB Page ID'].apply(lambda x: str(int(x)))

fb_factor = fb_factor.rename(columns={'FB Service Code': 'ServiceID',
                                      'FB Page ID': 'Linked FB Account',
                                      'Week Number': 'WeekNumber_finYear'})
fb_factor = gnl_expander(fb_factor)
twitter_df = rescaled_df.merge(fb_factor[['Linked FB Account', 'WeekNumber_finYear', 'Factor']], 
                               on=['Linked FB Account', 'WeekNumber_finYear'], 
                               how='left', indicator=True)
#print(f"1: {twitter_df.columns}")
print(f"1: {twitter_df.shape}")

done = twitter_df[twitter_df['_merge'] == 'both'].drop(columns=['_merge'])
need_fix = twitter_df[twitter_df['_merge'] == 'left_only'].drop(columns=['_merge', 'Factor'])

fb_factor_service = fb_factor.groupby(['ServiceID', 'WeekNumber_finYear'])['Factor'].mean().reset_index()
fixed = need_fix.merge(fb_factor_service, on=['ServiceID', 'WeekNumber_finYear'], how='left')
twitter_df = pd.concat([done, fixed])
#print(f"2: {twitter_df.columns}")
print(f"2: {twitter_df.shape}")

fb_videoMetric = pd.read_excel("../data/stale/FB Video Metrics.xlsx")
fb_videoMetric = fb_videoMetric.rename(columns={
    'FB Page ID': 'Linked FB Account',
    'GAM Week Number': 'WeekNumber_finYear',
    'FB Service Code': 'ServiceID'
})
fb_videoMetric = gnl_expander(fb_videoMetric)
fb_videoMetric['Linked FB Account'] = fb_videoMetric['Linked FB Account'].apply(
    lambda x: str(int(x)) if pd.notnull(x) and str(x).strip() != '' else ''
)

twitter_df = twitter_df.merge(fb_videoMetric, 
                              on=['Linked FB Account', 'WeekNumber_finYear', 'ServiceID'], 
                              how='left', indicator=True)
#print(f"3: {twitter_df.columns}")
print(f"3: {twitter_df.shape}")

done = twitter_df[twitter_df['_merge'] == 'both'].drop(columns=['_merge'])
need_fix = twitter_df[twitter_df['_merge'] == 'left_only'].drop(columns=['_merge'])\
        .drop(columns=['30 VTR%', '30 Sec Views per Viewer'])
unmatched_vtr = fb_videoMetric.groupby(['ServiceID', 'WeekNumber_finYear'])[['30 VTR%', '30 Sec Views per Viewer']].mean().reset_index()
fixed = need_fix.merge(unmatched_vtr, on=['ServiceID', 'WeekNumber_finYear'], how='left')

twitter_df = pd.concat([done, fixed])
#print(f"4: {fixed.columns}")
print(f"4: {twitter_df.shape}")


# In[13]:


twitter_df['video_video_views'] = twitter_df['video_video_views'].fillna(0)
twitter_df['30 sec viewers'] = twitter_df['30 VTR%']*twitter_df['video_video_views']/(twitter_df['30 Sec Views per Viewer']*1.1)
twitter_df['temp'] = twitter_df['tweet_engagements']/twitter_df['Factor']
twitter_df['Twitter Engaged Users'] = np.where(twitter_df['temp']>twitter_df['30 sec viewers'], 
                                                  twitter_df['temp']+twitter_df['30 sec viewers']*0.0733,
                                                    twitter_df['30 sec viewers']+twitter_df['temp']*0.2042)

twitter_df['uv_by_country'] = twitter_df["Twitter Engaged Users"]*twitter_df["engagement_%"]


# In[14]:


print(twitter_df.shape)
twitter_df = twitter_df.dropna(subset='uv_by_country')
print(twitter_df.shape)
cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
twitter_df[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv",
                        index=None)


# In[ ]:





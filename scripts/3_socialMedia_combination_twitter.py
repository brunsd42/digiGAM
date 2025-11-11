#!/usr/bin/env python
# coding: utf-8

# In[29]:


platformID = 'TWI'


# ## Status 
# twitter data is currently ingested and the process of adding facebook factors to it has been started, however the output files still show significant differences to minnie's dataset. 
# next step here is retesting the input files and then step by step through the combination process. 
# 
# twitter business unit and aggregated services is currently calculated by using minnie's dataset (helper/tw_minnie_preBU.csv)

# In[30]:


from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2


# ## import helper

# In[31]:


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
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[32]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
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


# # temporary fix

# In[33]:


'''cols_rename = {'Week Number': 'WeekNumber_finYear', 
               'Country Code': 'PlaceID', 
               'Service Code': 'ServiceID', 
               'TW Account ID': 'Channel ID', 
               'Twitter Engaged Users by Country': 'uv_by_country'}

full_df = pd.read_csv(f"helper/tw_minnie_preBU.csv")
display(full_df.sample())
full_df = full_df.rename(columns=cols_rename)[cols_rename.values()]
full_df['PlatformID'] = platformID
# w/c	PlaceID	ServiceID	Channel ID	uv_by_country
full_df = full_df.merge(week_tester[['WeekNumber_finYear', 'w/c']], 
                        on='WeekNumber_finYear', how='left').drop(columns=['WeekNumber_finYear'])
display(full_df.sample())

full_df.to_csv(f"../data/processed/{platformID}/minnie_uniqueViewer_country.csv", index=None)'''


# # ingestion

# In[34]:


tw_activity_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT.csv")
tw_activity_df['week_ending'] = pd.to_datetime(tw_activity_df['week_ending'])
tw_activity_df.columns


# In[35]:


tw_activity_df = tw_activity_df.merge(week_tester[['week_ending', 'WeekNumber_finYear', 'w/c']], 
                                      on=['week_ending'], how='left')

tw_activity_df['tw_account_id'] = tw_activity_df['tw_account_id'].apply(lambda x: str(int(x)))

tw_activity_df.head()


# In[36]:


tw_country_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country.csv",
                            low_memory=False)

tw_country_df = tw_country_df.rename(columns={
    'TW Account ID': 'tw_account_id',
    'Week Number': 'WeekNumber_finYear'})

tw_country_df = tw_country_df.drop_duplicates().drop(columns=['Week Commencing', 'Actual Week'])

tw_country_df['tw_account_id'] = tw_country_df['tw_account_id'].fillna('').apply(lambda x: str(int(x)))

tw_country_df['TW Linked FB account'] = tw_country_df['TW Linked FB account'].apply(
    lambda x: str(int(x)) if pd.notnull(x) and str(x).strip() != '' else ''
)

# TODO
# test uniqueness - account / week 
tw_country_df.sample()


# In[37]:


tw_country_df[(tw_country_df['tw_account_id'] == 146478129) &
            #(tw_country_df['Country'] == 'Argentina') &
            (tw_country_df['WeekNumber_finYear'] == 1)]


# # combine 

# In[38]:


tw_activity_country = tw_activity_df.merge(tw_country_df, on=['tw_account_id', 'WeekNumber_finYear'], 
                                           how='left', indicator=True)

test_functions.test_inner_join(tw_activity_df, tw_country_df, ['tw_account_id', 'WeekNumber_finYear'], 
                               '1_TW_10', test_step='joining activity & country - first')


# In[39]:


left_over = tw_activity_country[tw_activity_country._merge == 'left_only'].drop(columns='_merge')
final_1 = tw_activity_country[tw_activity_country._merge != 'left_only'].drop(columns='_merge')

# groupby 
grouped_df = tw_country_df.groupby(['tw_account_id', 'TW Account Name', 'TW Account Handle', 
                                    'TW Service Code', 'TW Linked FB account', 
                                    'TW Studios Exc UK', 'Country']).agg({
    'Weekly Engagements': 'mean',
    'Engagement %': 'mean',
    'Weekly Video Views': 'mean'
}).reset_index()

# twitter_activity_metadata.columns: to keep the columns from the initial merge and loose all those that are empty anyway
final_2 = left_over[tw_activity_df.columns].merge(grouped_df, on='tw_account_id', how='left')#.drop(columns='_merge')
test_functions.test_inner_join(left_over, grouped_df, ['tw_account_id'], 
                               '1_TW_11', test_step='joining activity & country - second')


# In[40]:


cols = ['tw_account_id', 'TW Account Name', 'TW Account Handle',
        'Weekly Engagements', 'Weekly Video Views',
        'w/c', 'WeekNumber_finYear', 'TW Linked FB account', 'TW Studios Exc UK', 
        'Country', 'Engagement %', 'TW Service Code' ]
final_df = pd.concat([final_1, final_2])[cols]

final_df['Weekly Engagements'] = np.where(final_df['Weekly Engagements']<0, 0, 
                                          final_df['Weekly Engagements'])

# TODO: find out why there are so many duplicates
print(final_df.shape)
print(final_df.drop_duplicates().shape)

final_df = final_df.rename(columns={'TW Service Code': 'ServiceID', }).drop_duplicates()
# file is used to compare to minnie's dataset:
final_df.to_csv(f"../data/processed/{platformID}/temp_{gam_info['file_timeinfo']}_metric_country.csv", index=None)


# In[41]:


# handle if country == 'other'
regular_country = final_df[final_df['Country'] != 'Other']
regular_country_100 = final_df[final_df['Country'] != 'Other']
regular_country_100 = regular_country_100.groupby(['tw_account_id', 'WeekNumber_finYear'])['Engagement %'].sum().reset_index()

rescaled_df = regular_country.merge(regular_country_100, on=['tw_account_id', 'WeekNumber_finYear'], how='left',
                                    suffixes=['_', 'newTotal'])

rescaled_df['engagement_%'] = rescaled_df["Engagement %_"]/rescaled_df["Engagement %newTotal"]


# ## facebook factor 

# In[42]:


fb_factor = pd.read_excel("../helper/FB Factor for IG and TW.xlsx").drop_duplicates()
fb_factor['FB Page ID'] = fb_factor['FB Page ID'].apply(lambda x: str(int(x)))

fb_factor = fb_factor.rename(columns={'FB Service Code': 'ServiceID',
                                      'FB Page ID': 'TW Linked FB account',
                                      'Week Number': 'WeekNumber_finYear'})
twitter_df = rescaled_df.merge(fb_factor[['TW Linked FB account', 'WeekNumber_finYear', 'Factor']], 
                               on=['TW Linked FB account', 'WeekNumber_finYear'], 
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

fb_videoMetric = pd.read_excel("../helper/FB Video Metrics.xlsx")
fb_videoMetric = fb_videoMetric.rename(columns={
    'FB Page ID': 'TW Linked FB account',
    'GAM Week Number': 'WeekNumber_finYear',
    'FB Service Code': 'ServiceID'
})

fb_videoMetric['TW Linked FB account'] = fb_videoMetric['TW Linked FB account'].apply(
    lambda x: str(int(x)) if pd.notnull(x) and str(x).strip() != '' else ''
)

twitter_df = twitter_df.merge(fb_videoMetric, 
                              on=['TW Linked FB account', 'WeekNumber_finYear', 'ServiceID'], 
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


# In[43]:


twitter_df['Weekly Video Views'] = twitter_df['Weekly Video Views'].fillna(0)
twitter_df['30 sec viewers'] = twitter_df['30 VTR%']*twitter_df['Weekly Video Views']/(twitter_df['30 Sec Views per Viewer']*1.1)
twitter_df['temp'] = twitter_df['Weekly Engagements']/twitter_df['Factor']
twitter_df['Twitter Engaged Users'] = np.where(twitter_df['temp']>twitter_df['30 sec viewers'], 
                                                  twitter_df['temp']+twitter_df['30 sec viewers']*0.0733,
                                                    twitter_df['30 sec viewers']+twitter_df['temp']*0.2042)

twitter_df['uv_by_country'] = twitter_df["Twitter Engaged Users"]*twitter_df["engagement_%"]


# In[44]:


twitter_df['Country'] = twitter_df['Country'].fillna('Unknown')
twitter_df_clean = twitter_df.rename(columns={'Country': 'TWI_CountryName'})
twitter_df_clean = twitter_df_clean.merge(country_codes[['TWI_CountryName', 'PlaceID', gam_info['population_column']]], 
                                                                               on='TWI_CountryName', how='left', indicator=True)


# In[45]:


print(twitter_df_clean._merge.value_counts())


# In[46]:


twitter_df_clean = twitter_df_clean.rename(columns={
    'tw_account_id': 'Channel ID',
})
cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country', gam_info['population_column']]
twitter_df_clean[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv",
                        index=None)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[30]:


platformID = 'YT-'


# ## import libraries

# In[31]:


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

# In[32]:


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
# Now import your modules 
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[33]:


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

socialmedia_accounts = socialmedia_accounts[(socialmedia_accounts['PlatformID'] == platformID)
                                            & 
                                            (socialmedia_accounts['Status'] == 'active')]
socialmedia_accounts = socialmedia_accounts.rename(columns={'Excluding UK': 'Channel Group'})

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)
socialmedia_accounts.sample()


# # Unique Viewers

# ## Ingestions 

# ### automated extracts

# In[34]:


main_path = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_export/"
#main_path = f"../../../../Research Projects/GAM/Digital GAM/2025/Social Media/"

# Dynamically get all folders in the main_path
folder_paths = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]

### TESTING input files ### 
test_functions.youtube_test_input_files('1_YT_1', folder_paths, main_path, week_tester, test_step='testing automated extracts')


# In[35]:


# ingest files
output_csv_path = f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_zipfiles_BBC World Service.csv"

# Check if the output CSV already exists
if os.path.exists(output_csv_path):
    combined_df = pd.read_csv(output_csv_path)
else:
    combined_df = pd.DataFrame()

# Dynamically get all folders in the main_path
folder_paths = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]
print('if files have been previously extracted the unzipping will be skipped')
                
for folder in folder_paths:    
    for file_name in tqdm(os.listdir(main_path+folder)):
        if file_name.endswith('.zip'):
                # Check if the file has already been processed
                if 'source_path' in combined_df.columns and (main_path+folder + file_name) in combined_df['source_path'].values:
                    #print(f"Skipping {file_name} as it has already been processed.")
                    continue

                # TODO next year: advertisement is identified as in each folder is a subfolder total
                # and a second subfolder called Advertisment - to process advertisement that has to be added
                # as a flag to the individual exports here 
                with zipfile.ZipFile(os.path.join(main_path+folder, file_name), 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if 'Table data.csv' in member:
                            with zip_ref.open(member) as file:
                                df = pd.read_csv(file)
                            
                            # Extract start date, end date, and content manager from the file name
                            match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2}) (.+)', file_name)
                            if match:
                                df['w/c'] = pd.to_datetime(match.group(1), format='%Y-%m-%d')
                                df['Channel Group'] = match.group(3)
                                df['source_path'] = main_path+folder + file_name
    
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                            combined_df.to_csv(output_csv_path, index=False)

# processing 
combined_df['w/c'] = pd.to_datetime(combined_df['w/c'], format='ISO8601')
combined_df['w/c'] = combined_df['w/c'] - pd.to_timedelta(combined_df['w/c'].dt.weekday, unit='D')

combined_df['week_ending'] = combined_df['w/c'] + pd.to_timedelta(6 - combined_df['w/c'].dt.weekday, unit='D')

combined_df['Channel Group'] = combined_df['Channel Group'].str.replace('.zip', '')
combined_df = combined_df.rename(columns={'Channel': 'Channel ID'})

# TODO: confirm what to do with the total (so far it's excluded at the inner join with social media accounts)
combined_df = combined_df.loc[combined_df['Channel ID'] != 'Total']

# confirm dtypes 
combined_df.loc[:, 'Unique viewers'] = pd.to_numeric(combined_df['Unique viewers'], errors='raise').astype('Int64')
combined_df.loc[:, 'Views'] = pd.to_numeric(combined_df['Views'].fillna(0), errors='raise').astype('Int64')
combined_df.loc[:, 'Watch time (hours)'] = pd.to_numeric(combined_df['Watch time (hours)'], errors='raise')


# In[36]:


# TODO: find out from Minnie why Impressions and Impression click-through rate (%) is not in this dataset -> can be ignored
try:
    combined_df.loc[:, 'Impressions'] = combined_df['Impressions'].fillna(0)
    combined_df.loc[:, 'Impressions'] = pd.to_numeric(combined_df['Impressions'], errors='raise').astype('Int64')
except:
    print('could not change type of impressions - col does not exist and were created')

try:
    combined_df.loc[:, 'Impression click-through rate (%)'] = pd.to_numeric(combined_df['Impression click-through rate (%)'], errors='raise')
except:
    print('could not change type of impressions click through rate - col does not exist and was created')
    combined_df.loc[:, 'Impression click-through rate (%)'] = 0
combined_df.sample()


# ### manual extracts
# Media Action and channel by channel exports 

# In[37]:


#TODO: review with minnie for individual exports
#because it contains geography or should we use table instead?

path = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_manual/"
dataframes = []

for filename in os.listdir(path):
    if filename.endswith('.xlsx'):  # Assuming the files are excel files
        
        try:
            file_path = os.path.join(path, filename)
            df = pd.read_excel(file_path, sheet_name='Totals')
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

# Group by Geography, FromDate, ToDate, and filename to sum Views
media_action_df = media_action_df.groupby(['w/c', 'week_ending', 'Channel ID', 'Channel title', 'source_path']).agg({'Views': 'sum'}).reset_index()
media_action_df['Channel Group'] = 'BBC Media Action'

channel_ids = {'Aksi Kita Indonesia': 'aksikitaindo', }
media_action_df['Channel ID'] = media_action_df['Channel ID'].replace(channel_ids)

media_action_df['Unique viewers'] = media_action_df['Views'] / gam_info['overlap_viewer_uniqueViever']


# In[38]:


gam_info['overlap_viewer_uniqueViever']


# ### combine CMS & non CMS

# In[39]:


full_uv_df = pd.concat([combined_df, media_action_df])
print(full_uv_df.shape) #(7841, 14)

# add service & service code info 
youtube_uv = full_uv_df.merge(socialmedia_accounts[['Channel ID', 'Channel Name',  'Service', 'ServiceID']], 
                              on='Channel ID' , how='left')

youtube_uv['Unique viewers'] = youtube_uv['Unique viewers'].fillna(0)
youtube_uv.drop(columns=['week_ending'], inplace=True)

# TODO add test to ensure no data is lost with these 
#      (and keep on left ot make sure we never loose data)
#youtube_uv = youtube_uv.merge(week_tester[['week_ending', 'w/c']], on='week_ending', how='left', indicator=True)
#print(youtube_uv._merge.value_counts())



# ## Test 

# In[40]:


################################### Testing ################################### 
test_step = 'combine CMS & non CMS'
# test accounts
test_functions.test_filter_elements_returned(youtube_uv, channel_ids, 'Channel ID', "1_YT_2", test_step)
# test weeks 
test_functions.test_weeks_presence_per_account('w/c', 'Channel ID', youtube_uv, week_tester, "1_YT_3", test_step)
# test duplicates
cols= ['Channel ID', 'Channel title', 'Channel Group', 'w/c',]
test_functions.test_duplicates(youtube_uv, cols, '1_YT_4', test_step)

test_functions.test_merge_row_count(youtube_uv, full_uv_df, '1_YT_5', test_step)

################################### Testing ################################### 

youtube_uv.sample()


# In[41]:


youtube_uv[youtube_uv['Channel ID'] == 'UCyL1hGLVGqeZ1ak3DJeik7Q']


# ## Storing

# In[42]:


cols = ['Channel ID', 'Channel Name', 'ServiceID', 'Channel Group',
        'Channel title', 'Unique viewers', 'w/c']
youtube_uv = youtube_uv[cols]

# clean cols 
youtube_uv['ServiceID'] = youtube_uv['ServiceID'].str.strip().fillna('')
youtube_uv['Channel ID'] = youtube_uv['Channel ID'].str.strip().fillna('')
youtube_uv['Unique viewers'] = youtube_uv['Unique viewers'].fillna(0)
youtube_uv['w/c'] = pd.to_datetime(youtube_uv['w/c'])

youtube_uv.to_csv(f"../data/processed/{platformID}/_{gam_info['file_timeinfo']}_uniqueViewer_withAds.csv", 
                         index=None)


# ## remove Advertising 

# In[43]:


# read in ad dataset
cols = ['Channel', 'Week', '% reach to be removed']
youtube_ads = pd.read_excel(f"../data/raw/{platformID}/YouTube_advertising.xlsx", usecols=cols)
youtube_ads.rename(columns={'Channel': 'Channel ID', 
                            'Week': 'w/c'}, inplace=True)

# merge datasetsa
youtube_uv_withAds = youtube_uv.merge(youtube_ads, on=['Channel ID', 'w/c'], how='left')
youtube_uv_withAds['% reach to be removed'] = youtube_uv_withAds['% reach to be removed'].fillna(0)

# subset youtube_uv dataset 
youtube_uv_organic = youtube_uv_withAds.copy()
youtube_uv_organic['Unique viewers'] -= youtube_uv_organic['Unique viewers'].mul(youtube_uv_organic['% reach to be removed'])
youtube_uv_organic.drop(columns=['% reach to be removed'], inplace=True)
youtube_uv_organic.head()


# ## Testing 

# In[44]:


# Get a rough estimate of channel average UV and sort descending by average UV
channel_avg_uv = youtube_uv_organic.groupby(['Channel ID', 'ServiceID'])['Unique viewers'].mean()\
                            .reset_index(name='average_UV')\
                            .sort_values(by='average_UV', ascending=False)

## TODO make heatmap


# In[45]:


# Calculate the sum of unique viewers for each YT Service Code and Week Number
sum_uv = youtube_uv_organic.groupby(['ServiceID', 'w/c'])['Unique viewers'].sum()\
                               .reset_index(name='sum_UV')

# Calculate the average of unique viewers for each YT Service Code
avg_uv = youtube_uv_organic.groupby(['ServiceID'])['Unique viewers'].mean()\
                    .reset_index(name='average_UV')\
                    .sort_values(by='average_UV', ascending=False)

## TODO make heatmap


# # storing dataset

# In[46]:


youtube_uv_organic.to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer.csv", 
                         index=None)


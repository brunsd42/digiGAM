#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
pd.set_option('display.max_colwidth', None)


import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import urllib.parse
import json

import os
from tqdm import tqdm


# ## import helper

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
from security_config import api_key

import test_functions
import functions


# In[3]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)
country_codes = country_codes.rename(columns={'ATI': 'geo_country'})

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
#week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# site info - with api query
site_info = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='Site_API').drop(columns='no results')
site_info['Report No.'] = site_info['Report No.'].astype(str)
site_info = site_info[site_info['script'] == '1_site_ingestion']

# platform codes
platform_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='PlatformID')#[cols]

# service codes
service_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='ServiceID')#[cols]
service_codes = service_codes.rename(columns={'ATI (Level 2 site)': 'site_level2'})

# language service map 
service_language_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='Site_language')

# non js 
non_js_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='Site_NonJS')

# app
app_map = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='Site_App')


# # ingestion
# ## Chartbeat
# the idea is to get site reach also from chartbeat and have a comparison - if the values ranges widely it needs to be reviewed why and what the most accurate reach is.

# ## Piano

# In[4]:


test_functions.site_test_unique_entries(site_info, 'Report No.', '1_Site_1', 'initial api query list')


# In[5]:


i = 0
for index, row in site_info.iterrows():
    # TODO find Studios / WSL logic switch for API keys
    api_query = row['API']
    api_query_key = api_key[row['api_key']]
    report_no = row['Report No.']

    #print(convert_url_to_query(api_query, start, end))
    print(f"starting report no {report_no}")
    print(api_query)
    
    for jndex, row in week_tester.iterrows():
        week_number = row['WeekNumber_finYear']
        path = "../data/raw/site/piano_reports"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/{gam_info['file_timeinfo']}_reportNo{report_no}_weekNo{week_number}.csv"
        
        # Check if the file exists, if so, continue to the next iteration
        if os.path.exists(filename):
            continue
            
        print(f"... iteration {filename}")
        start = row['w/c'] # dtype object
        end = row['week_ending'] # dtype object

        if pd.to_datetime(end) > datetime.today():
            print('date in future')
            break
        # convert to api query 
        query = functions.convert_url_to_query(api_query, start, end)
        
        # run api query 
        temp = functions.api_call(query, api_query_key)
        
        temp['w/c'] = start
        temp['timestamp_queryRun'] = datetime.now().strftime('%y%m%d-%H%M')
        temp['API'] = api_query

        if temp.shape[0] == 0:
            temp = pd.DataFrame()
        
        temp.to_csv(filename, index=None)
        
    print(f"finished report no {report_no}")


# In[6]:


# test if more than 10000 rows are recorded to see if the pagination works 
    # yes it wokrs: great well done! 
    # no it doesn't: rerun all 10000 long queries
# Result: yes it works and also no report is larger than 200'000 (18k)


# # Analysis 

# ## Chartbeat vs Piano

# In[7]:


# build at home


# # Processing 

# In[8]:


filepath = f"../data/raw/site/piano_reports/"
all_files, empty_report_list = [], []
size = 0
for file in tqdm(os.listdir(filepath)):
    
    if (gam_info['file_timeinfo'] in file):
        temp= pd.read_csv(filepath+file)
        if len(temp.columns) == 3:
            empty_report_list.append(file)
        # measuring how many rows the largest file has
        if temp.shape[0] > size:
            size= temp.shape[0] 
        temp['filename'] = file
        parts = file.split('_')
        temp['Report No.'] = parts[1]
        temp['Report No.'] = temp['Report No.'].str.extract('(\d+)')[0]
        
        all_files.append(temp)

print(f"largest file is {size} rows long")
#empty_report_list.to_csv(f"../test/specific/{gam_info['file_timeinfo']}_empty_report_returns.csv")

combined_df = pd.concat(all_files)
if 'API' not in combined_df.columns:
    print('adding API')
    combined_df['API'] = ''
#combined_df['API'] = combined_df['API'].fillna(combined_df['api_query'])
#combined_df.drop(columns=['api_query'], inplace=True)
combined_df['w/c'] = pd.to_datetime(combined_df['w/c'] )


# In[9]:


# test all reports are there 
test_functions.test_inner_join(site_info, combined_df, ['Report No.', 'API'], 
                               '1_Site_2', 'adding report context info', focus='left')

# add report info
full_df = site_info.merge(combined_df, on=['Report No.', 'API'], how='inner', )
#print(full_df['Report No.'].unique())

# test all weeks are there 
test_functions.test_weeks_presence_per_account('w/c', 'Report No.', full_df, week_tester, 
                                               '1_Site_3', test_step='combining api returns')

# add week_lookup data
full_df = full_df.merge(week_tester[['YearGAE', 'WeekNumber_finYear', 'w/c']], on='w/c', how='left')
# excluded: 'API', 'timestamp_queryRun', 'filename', 'Year',
cols = ['Category', 'Report No.', 'Space', 'Description', 
        'YearGAE', 'WeekNumber_finYear', 'w/c',  
        'site_level2', 'geo_country', 'm_unique_visitors', 
        #'app_name', 
        'device_type', 'language', 'producer_nonjs']
full_df = full_df[cols]


# Specify the dtype option to avoid DtypeWarning for columns with mixed types
dtype_spec = {
    #'m_unique_visitors': int,
    'Report No.': str,
    'device_type': str,
    'app_name': str,
    'language': str,
    'producer_nonjs': str,
    'src': str
}

# Convert columns to the specified dtypes
for column, dtype in dtype_spec.items():
    if column in full_df.columns:
        full_df[column] = full_df[column].apply(lambda x: str(x) if pd.notnull(x) else '')

full_df.to_csv(f"../data/raw/site/{gam_info['file_timeinfo']}_rawDataFromPiano.csv", index=None)


# In[10]:


full_df.head()


# In[ ]:





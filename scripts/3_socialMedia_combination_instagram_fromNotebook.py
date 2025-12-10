#!/usr/bin/env python
# coding: utf-8

# In[7]:


platformID = 'INS'


# In[8]:


from datetime import datetime
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
import numpy as np
import os 

import psycopg2


# ## import helper 

# In[9]:


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
import test_functions 


# In[10]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
socialmedia_accounts = pd.read_excel("../helper/ins_account_lookup.xlsx")

### RUN TESTS
test_functions.test_lookup_files(country_codes, ['PlaceID'], [f"{platformID}_3_0", f"{platformID}_3_1", f"{platformID}_3_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3_3", f"{platformID}_3_4", f"{platformID}_3_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], [f"{platformID}_3_6", f"{platformID}_3_7", f"{platformID}_3_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")



# # ingest 

# In[11]:


engagements = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_engagements_final.csv",)
                                             
engagements['w/c'] = pd.to_datetime(engagements['w/c'])
engagements['Channel ID'] = engagements['Channel ID'].dropna().apply(lambda x: str(int(x)))
engagements.sample()


# In[12]:


engagements[(engagements['ServiceID'] == 'PER') & 
    (engagements['w/c'] == '2025-05-05') 
    #& (engagements['PlaceID'] == 'IRN')
    ]#['uv_by_country'].sum()


# In[13]:


country = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv",
                     keep_default_na=False)
country['w/c'] = pd.to_datetime(country['w/c'])
country['Channel ID'] = country['Channel ID'].dropna().apply(lambda x: str(int(x)))
country.sample()


# In[14]:


country_annual_avg = country.groupby(['Channel ID', 'Channel Name', 'PlaceID'])['country_%'].mean().reset_index()


# # combine 

# In[15]:


combined = engagements.merge(country, on=["Channel ID", "w/c"], how='left', indicator=True)
combined_inner = combined[combined['_merge'] == 'both'].drop(columns='_merge')
combined_left = combined[combined['_merge'] == 'left_only'].drop(columns='_merge')

left_matched = combined_left.merge(country_annual_avg, on="Channel ID", how='left', indicator=True)

cols_to_clean = ['Channel Name', 'PlaceID', 'country_%',]
for col in cols_to_clean:
    left_matched[f"{col}"] = left_matched[f"{col}_x"].fillna(left_matched[f"{col}_y"])
    left_matched = left_matched.drop(columns=[f"{col}_x", f"{col}_y"])

temp = left_matched[left_matched['_merge'] == 'left_only'].drop(columns='_merge')
temp = temp.merge(socialmedia_accounts[['Channel ID', 'IG Handle']], on='Channel ID', how='left')
missing_country_perc = pd.read_excel("../data/stale/missing ig countries.xlsx")
missing_country_perc['country code'] = missing_country_perc['country code'].str.upper()
missing_country_perc.rename(columns={'country code': 'PlaceID', 'Total': 'country_%'})
temp = temp.merge(missing_country_perc, on='IG Handle', how='inner')
cols = ['Channel ID', 'Channel Name', 'IG Handle', 
        'w/c', 'ServiceID',
        'plays', 'impressions',
        '30 view', 'IG Modelled Factor', 
        'PlaceID', 'engaged_reach', 'country_%']
temp = temp[cols]


# In[16]:


cols = ['Channel ID', 'Channel Name', 'IG Handle', 
       'w/c', 'ServiceID',  'plays',  'PlaceID', 'engaged_reach', 'country_%']
engagement_country = pd.concat([combined_inner, temp])[cols].rename(columns={'IG Engaged Persian Exception': 'IG Engaged Users'})



# In[17]:


to_clean_country = country_codes[['PlaceID', 'YT-_FBE_codes', gam_info['population_column']]]
clean_country = engagement_country.merge(to_clean_country, on='PlaceID', how='left', indicator=True)
print(clean_country.shape)
final_ig_data = clean_country.drop_duplicates(subset=['PlaceID', 'w/c', gam_info['population_column'],
                                                      'Channel ID', 'Channel Name'])
print(final_ig_data.shape)
final_ig_data['engaged_reach'] = final_ig_data['engaged_reach'].fillna(0)
final_ig_data['country_%'] = final_ig_data['country_%'].fillna(0)
final_ig_data['uv_by_country'] = final_ig_data['engaged_reach'] * final_ig_data['country_%']


# In[18]:


final_ig_data[(final_ig_data['ServiceID'] == 'PER') & 
    (final_ig_data['w/c'] == '2025-05-05') & 
    (final_ig_data['PlaceID'] == 'IRN')]#['uv_by_country'].sum()


# # store 

# In[19]:


print(final_ig_data.shape)
final_ig_data = final_ig_data.dropna(subset='uv_by_country')
print(final_ig_data.shape)

cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
final_ig_data[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv",
                     index=None)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[8]:


platformID = 'INS'


# In[9]:


from datetime import datetime
import pandas as pd
import numpy as np
import os 

import psycopg2


# ## import helper 

# In[10]:


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
import test_functions 


# In[11]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
socialmedia_accounts = pd.read_excel("../helper/ins_account_lookup.xlsx")


# # ingest 

# In[12]:


engagements = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_engagements_final.csv")
engagements.columns


# In[13]:


country = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_geog.csv")
country.columns


# In[14]:


country_annual_avg = country.groupby(['Channel ID', 'Channel Name', 
                                      'ig_metric_breakdown'])['country_%'].mean().reset_index()


# # combine 

# In[15]:


combined = engagements.merge(country, on=["Channel ID", "w/c"], how='left', indicator=True)
combined_inner = combined[combined['_merge'] == 'both'].drop(columns='_merge')
combined_left = combined[combined['_merge'] == 'left_only'].drop(columns='_merge')

left_matched = combined_left.merge(country_annual_avg, on="Channel ID", how='left', indicator=True)

cols_to_clean = ['Channel Name', 'ig_metric_breakdown', 'country_%',]
for col in cols_to_clean:
    left_matched[f"{col}"] = left_matched[f"{col}_x"].fillna(left_matched[f"{col}_y"])
    left_matched = left_matched.drop(columns=[f"{col}_x", f"{col}_y"])

temp = left_matched[left_matched['_merge'] == 'left_only'].drop(columns='_merge')
temp = temp.merge(socialmedia_accounts[['Channel ID', 'IG Handle']], on='Channel ID', how='left')
missing_country_perc = pd.read_excel("../../../../Research Projects/GAM/Digital GAM/2025/Social Media/data/missing ig countries.xlsx")
missing_country_perc['country code'] = missing_country_perc['country code'].str.upper()
missing_country_perc.rename(columns={'country code': 'ig_metric_breakdown', 'Total': 'country_%'})
temp = temp.merge(missing_country_perc, on='IG Handle', how='inner')
cols = ['Channel ID', 'Channel Name', 'IG Handle', 'ig_user_linked_fb_page_id', 
        'w/c', 'IG Account URL', 'ServiceID', 'IG studios exc uk',
        'weekly_media_engagements', 'saved', 'plays', 'daily_avg_reach', 'reach', 'impressions',
        '30 view', 'IG Modelled Factor', 'IG Engaged Users', 'IG Engaged Persian Exception',
        'ig_metric_breakdown', 'country_%']
temp = temp[cols]


# In[16]:


cols = ['Channel ID', 'Channel Name', 'IG Handle', 'ig_user_linked_fb_page_id',
       'w/c', 'IG Account URL', 'ServiceID', 'IG studios exc uk',
       'plays', 'daily_avg_reach', 'IG Engaged Persian Exception', 
        'ig_metric_breakdown', 'country_%']
engagement_country = pd.concat([combined_inner, temp])[cols].rename(columns={'IG Engaged Persian Exception': 'IG Engaged Users'})



# In[17]:


to_clean_country = country_codes[['PlaceID', 'YT-_FBE_codes', gam_info['population_column']]].rename(columns={'YT-_FBE_codes': 'ig_metric_breakdown'})
clean_country = engagement_country.merge(to_clean_country, on='ig_metric_breakdown', how='left', indicator=True)
print(clean_country.shape)
final_ig_data = clean_country.drop_duplicates(subset=['PlaceID', 'w/c', gam_info['population_column'],
                                                      'Channel ID', 'Channel Name', 'IG Account URL'])
print(final_ig_data.shape)
final_ig_data['uv_by_country'] = final_ig_data['IG Engaged Users'] * final_ig_data['country_%']


# # store 

# In[18]:


cols = ['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
final_ig_data[cols].to_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv",
                     index=None)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display

import pandas as pd 
pd.set_option('display.float_format', '{:.2f}'.format)

import numpy as np

import os
from tqdm import tqdm
from datetime import datetime, timedelta


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

import test_functions 
import functions 


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


platformID = 'WSC'
# country
pop_size_col = gam_info['population_column']

country_cols = ['PlaceID', pop_size_col]
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)[country_cols]
country_codes[pop_size_col] = (
    pd.to_numeric(country_codes[pop_size_col], errors='coerce')
    .fillna(1)
    .astype(int)
)
#########
# week 
week_cols = ['w/c', 'YearGAE', 'WeekNumber_finYear']
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period',)[week_cols]
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
today = pd.Timestamp.today()
last_monday = today - timedelta(days=today.weekday() + 7)
valid_weeks = week_tester[week_tester['w/c'] <= last_monday]
number_of_weeks = valid_weeks['WeekNumber_finYear'].max()
#########
service_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='ServiceID',)
service_hierarchy = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='Service Hierarchy',)
#########
platform_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='PlatformID',)
#########
overlap_SocWebOverlap = pd.read_excel("../data/stale/Final Overlaps 2021.xlsx", sheet_name='SocWebOverlap').drop(columns=['Population 2020']).drop_duplicates()
overlap_SocWebOverlap['PlaceID'] = overlap_SocWebOverlap['PlaceID'].replace('MYT', 'MAY').replace('WLF', 'WFI')
#overlap_SocWebOverlap = overlap_SocWebOverlap.merge(country_codes, on='PlaceID', how='left')


### RUN TESTS
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_6_0", f"{platformID}_6_1", f"{platformID}_6_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, week_cols, [f"{platformID}_6_3", f"{platformID}_6_4", f"{platformID}_6_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(service_tester, ['ServiceID'], [f"{platformID}_6_6", f"{platformID}_6_7", f"{platformID}_6_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")

test_functions.test_lookup_files(platform_tester, ['PlatformID'], [f"{platformID}_6_9", f"{platformID}_6_10", f"{platformID}_6_11"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# # functions 

# In[5]:


def compute_combined_reach(df, services, label, pop_size_col, country_codes, deal_with_zero=True, 
                          calc_type='sainsbury'):
    """
    Filters, merges, aggregates, and applies the Sainsbury formula to compute combined reach.

    Parameters:
    df (pd.DataFrame): Source DataFrame with weekly reach data.
    services (list): List of ServiceIDs to include.
    label (str): Label to assign to the resulting ServiceID.
    pop_size_col (str): Column name for population size.
    country_codes (pd.DataFrame): Mapping DataFrame for PlaceID enrichment.
    deal_with_zero (bool): Whether to apply shortcut logic in the formula.

    Returns:
    pd.DataFrame: Aggregated and transformed DataFrame with combined reach.
    """
    filtered_df = df[df['ServiceID'].isin(services)].merge(country_codes, on='PlaceID', how='left')
    
    pivot_df = pd.crosstab(
        index=[filtered_df['PlaceID'], filtered_df[pop_size_col], 
               filtered_df['w/c']],
        columns=filtered_df['ServiceID'],
        values=filtered_df['Reach'],
        aggfunc='sum'
    ).reset_index().fillna(0)

    if calc_type == 'add':
        pivot_df['Reach'] = pivot_df[services].sum(axis=1)
    elif calc_type == 'sainsbury':
        pivot_df = functions.sainsbury_formula(pivot_df, pop_size_col, services, 
                                               'Reach', deal_with_zero=deal_with_zero)
        
    else: 
        print('error')
        
    pivot_df['ServiceID'] = label
    return pivot_df[['w/c', 'ServiceID', 'PlaceID', 'Reach']]


# # ingestion 
# 
# ## workflow 5

# In[6]:


base_folder = "../data/singlePlatform/"
singlePlatform_df_list = []

valid_platforms = set(platform_tester['PlatformID']) 
valid_services = set(service_tester['ServiceID']) | {'WSL'}

# Traverse platform folders
for platform_id in os.listdir(base_folder):
    platform_path = os.path.join(base_folder, platform_id)
    if not os.path.isdir(platform_path) or platform_id not in valid_platforms:
        continue

    # Traverse weekly subfolders
    for weekly_folder in os.listdir(platform_path):
        weekly_path = os.path.join(platform_path, weekly_folder)
        if not os.path.isdir(weekly_path):
            continue

        # Traverse files in weekly subfolder
        for file in tqdm(os.listdir(weekly_path), desc=f"{platform_id}/{weekly_folder}"):
            if file == ".DS_Store" or "podcast" in file.lower() or "site" in file.lower():
                continue

            # Example: GAM2025_WEEKLY_FBE_EN2byCountry.xlsx
            try:
                name_parts = file.replace("byCountry.xlsx", "").split("_")
                if len(name_parts) >= 4:
                    platform_id_in_file = name_parts[2]
                    service_id = name_parts[3]

                    if (platform_id_in_file == platform_id) and (service_id in valid_services):
                        file_path = os.path.join(weekly_path, file)
                        temp = pd.read_excel(file_path)
                        temp['w/c'] = pd.to_datetime(temp['w/c'])
                        temp['source'] = file
                        temp['PlatformID'] = platform_id
                        #temp['ServiceID'] = service_id
                        temp['WeekFolder'] = weekly_folder
                        singlePlatform_df_list.append(temp)
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Combine all valid dataframes
singlePlatform_df = pd.concat(singlePlatform_df_list, ignore_index=True)


# In[7]:


test_step = 'reading in all single platform values'
# test that reach is >=0
test_functions.test_non_null_and_positive(singlePlatform_df, ['Reach'], 
                                          test_number=f"{platformID}_6_12", test_step=test_step)
# test that all the combinations are unique (Service, Country, Week, Platform)
test_functions.test_duplicates(singlePlatform_df, ['ServiceID', 'PlaceID', 'PlatformID', 'w/c'], 
                               test_number=f"{platformID}_6_13", test_step=test_step)
# test that all platofrms are there and for each platform all services 
test_functions.test_filter_elements_returned(singlePlatform_df, service_tester['ServiceID'].unique(), 'ServiceID', 
                                             test_number=f"{platformID}_6_14", test_step=test_step)
test_functions.test_filter_elements_returned(singlePlatform_df, country_codes['PlaceID'].unique(), 'PlaceID', 
                                             test_number=f"{platformID}_6_15", test_step=test_step)
test_functions.test_filter_elements_returned(singlePlatform_df, platform_tester['PlatformID'].unique(), 'PlatformID', 
                                             test_number=f"{platformID}_6_16", test_step=test_step)
test_functions.test_filter_elements_returned(singlePlatform_df, week_tester['w/c'].unique(), 'w/c', 
                                             test_number=f"{platformID}_6_17", test_step=test_step)


# In[8]:


singlePlatform_df.to_csv(f"../data/combinePlatforms/social_media_data_{gam_info['file_timeinfo']}_platform_weekly.csv")


# # processing 

# ## test columns (& remove total )

# In[9]:


annual_singlePlatform_df = singlePlatform_df.groupby(['PlaceID', 'ServiceID', 'PlatformID'])['Reach'].sum().reset_index()
# Calculate weeks per ServiceID
weeks_per_service = singlePlatform_df.groupby(['ServiceID', 'PlatformID'])['w/c'].nunique()

# Map weeks_count using the group keys
annual_singlePlatform_df['weeks_count'] = annual_singlePlatform_df.apply(
    lambda row: weeks_per_service.to_dict().get((row['ServiceID'], row['PlatformID']), 0),
    axis=1)

# if less than 12 weeks active 

# Apply conditional division
annual_singlePlatform_df['Reach'] = annual_singlePlatform_df.apply(
    lambda row: row['Reach'] / (number_of_weeks if row['weeks_count'] >= 12 else row['weeks_count']),
    axis=1
)

annual_singlePlatform_df.to_csv(f"../data/combinePlatforms/social_media_data_{gam_info['file_timeinfo']}_platform_annual.csv")


# ## workflow 6

# In[10]:


full_service_df = pd.crosstab(
        index=[ singlePlatform_df['PlaceID'], 
                singlePlatform_df['w/c'], 
                singlePlatform_df['ServiceID']],
        columns=singlePlatform_df['PlatformID'],
        values= singlePlatform_df['Reach'],
        aggfunc='sum'
    ).reset_index().fillna(0)

full_service_df = full_service_df.merge(country_codes, on='PlaceID', how='left', )
full_service_df.head()
cols = full_service_df.columns


# In[11]:


'''
full_service_df[
    (full_service_df['PlaceID'] == 'POL') &
    (full_service_df['ServiceID'].isin(ax2_services)) &
    #(ax2_df_raw['PlatformID'].isin(['INS', 'WSC'])) 
    (full_service_df['w/c'] == '2025-09-22')
    ]'''


# ### WSL

# In[12]:


# remove MA / WOR and agg services 
exclude_ids = ['WOR', 'MA-', 'AXE',
               'ENG', 'EN2', 'ENW', 
               'ANW', 'TOT', 'AX2', 'ANY', 'ALL', ]
weekly_ws_df = full_service_df[~full_service_df['ServiceID'].isin(exclude_ids)]

# add overlaps
# TODO filter columns in 
cols = [#'Tapestry Market', 'Country Name', 
        'PlaceID', 'FB & YT Factor', 'Own Web & Social Factor', 'Web', 'Facebook Incremental',
        'YouTube Incremental', 'Social Incremental if YouTube bigger',
        'Social Incremental if Facebook bigger', 'Social Incremental',
        '% Twitter', '% Instagram', '% socialdedup Factor',
        #'Unnamed: 15', 'Unnamed: 16'
]
weekly_ws_df = weekly_ws_df.merge(overlap_SocWebOverlap[cols], on='PlaceID', how='left', indicator=True)

# Define expected platform columns
platform_cols = ['FBE', 'INS', 'TWI', 'YT-', 'TTK', 'TEL']
                
# Add missing columns with default value 0
for col in platform_cols:
    if col not in weekly_ws_df.columns:
        print(f'{col} missing!')
        weekly_ws_df[col] = 0
        
# Step 1: Calculate Max Reach
weekly_ws_df['Max Reach'] = weekly_ws_df[platform_cols].max(axis=1)


# In[13]:


# Step 2: Identify Max Platform
def get_max_platform(row):
    if row['Max Reach'] == row['FBE']:
        return 'Facebook'
    elif row['Max Reach'] == row['YT-']:
        return 'YouTube'
    elif row['Max Reach'] == row['TWI']:
        return 'Twitter'
    elif row['Max Reach'] == row['TTK']:
        return 'Tiktok'
    else:
        return 'Instagram'

weekly_ws_df['Max Platform'] = weekly_ws_df.apply(get_max_platform, axis=1)
ttk_increment = 0.3

# Step 3: Calculate WSC1
def calculate_wsc1(row):
    if row['Max Platform'] == 'Facebook':
        return (row['FBE'] + row['YT-'] * row['YouTube Incremental'] +
                row['INS'] * row['% Instagram'] +
                row['TWI'] * row['% Twitter'] +
                ttk_increment * row['TTK'])
    elif row['Max Platform'] == 'YouTube':
        return (row['YT-'] + row['FBE'] * row['Facebook Incremental'] +
                row['INS'] * row['% Instagram'] +
                row['TWI'] * row['% Twitter'] +
                ttk_increment * row['TTK'])
    elif row['Max Platform'] == 'Instagram':
        return (row['INS'] + row['YT-'] * row['YouTube Incremental'] +
                row['FBE']  * row['Facebook Incremental'] +
                ttk_increment * row['TTK'])
    elif row['Max Platform'] == 'Tiktok':
        return (row['TTK'] + row['YT-'] * row['YouTube Incremental'] +
                row['INS'] * row['% Instagram'] +
                row['TWI'] * row['% Twitter'] +
                row['FBE'] * row['Facebook Incremental'])
    else:  # Twitter
        return (row['TWI'] + row['YT-'] * row['YouTube Incremental'] +
                row['INS'] * row['% Instagram'] +
                row['FBE'] * row['Facebook Incremental'])

weekly_ws_df[f'{platformID}1'] = weekly_ws_df.apply(calculate_wsc1, axis=1)

# Ensure all required columns exist, fill missing ones with 0
required_cols = ['FBE', 'YT-', 'INS', 'TWI', 'TTK', 'WEI', 'TEL', '% socialdedup Factor']
for col in required_cols:
    if col not in weekly_ws_df.columns:
        print(f'{col} missing!')
        weekly_ws_df[col] = 0

# Calculate WSC2
weekly_ws_df[f'{platformID}2'] = (
    (weekly_ws_df['FBE'] + weekly_ws_df['YT-'] + weekly_ws_df['INS'] + weekly_ws_df['TWI'] + weekly_ws_df['TTK']) 
    * weekly_ws_df['% socialdedup Factor']
)
weekly_ws_df.to_csv('../test/wsc_calculation_details.csv', index=None)


# In[14]:


'''weekly_ws_df[
    (weekly_ws_df['PlaceID'] == 'POL') &
    (weekly_ws_df['ServiceID'].isin(ax2_services)) &
    #(ax2_df_raw['PlatformID'].isin(['INS', 'WSC'])) 
    (weekly_ws_df['w/c'] == '2025-09-22')
    ]'''


# In[15]:


# Ensure all required columns exist
required_cols = [f'{platformID}1', f'{platformID}2', 'FBE', 'YT-', 'INS', 'TWI', 'TTK', 'WEI', 'TEL']
for col in required_cols:
    if col not in weekly_ws_df.columns:
        print(f'{col} missing!')
        weekly_ws_df[col] = 0

# Compute WSC Final
def compute_wsc_final(row):
    wsc1 = row[f'{platformID}1']
    wsc2 = row[f'{platformID}2']
    if (
        wsc2 < wsc1 or
        wsc2 < row['FBE'] or
        wsc2 < row['YT-'] or
        wsc2 < row['INS'] or
        wsc2 < row['TWI'] or
        wsc2 < row['TTK']
    ):
        return wsc1
    else:
        return wsc2

weekly_ws_df['Reach'] = weekly_ws_df.apply(compute_wsc_final, axis=1)
weekly_ws_df = weekly_ws_df[['w/c', 'PlaceID', 'ServiceID', 'Reach']]
weekly_ws_df.head()


# ### MA & Studios

# In[16]:


ma_wor_df = full_service_df[full_service_df['ServiceID'].isin(['WOR', 'MA-'])]
required_cols = ['FBE', 'YT-', 'INS', 'TWI', 'TTK', 'WEI', 'TEL']
for col in required_cols:
    if col not in ma_wor_df.columns:
        print(f'{col} missing!')
        ma_wor_df[col] = 0

ma_wor_df = functions.sainsbury_formula(ma_wor_df, pop_size_col, 
                                        required_cols, 
                                        'Reach')

weekly_ma_wor_df = ma_wor_df[['w/c', 'PlaceID', 'ServiceID', 'Reach']]
weekly_ma_wor_df.head()


# prep the aggregate calculation

# In[17]:


weekly_df = pd.concat([weekly_ws_df, weekly_ma_wor_df])


# ### ENW

# In[18]:


# Usage
enw_services = ['FOA', 'WSE']
enw_df = compute_combined_reach(weekly_df, enw_services, 'ENW', pop_size_col, country_codes)
enw_df.head()


# ### ENG

# In[19]:


# Usage
eng_services = ['GNL', 'WSE']
eng_df = compute_combined_reach(weekly_df, eng_services, 'ENG', pop_size_col, country_codes)
eng_df.head()


# ### EN2

# In[20]:


en2_services = ['ENG', 'WOR']
en2_df = compute_combined_reach(pd.concat([weekly_df, eng_df]), en2_services, 'EN2', pop_size_col, country_codes)


# ### AX2

# In[21]:


cols = ['PlaceID', 'digiGAM_FOA_WT-']
africa_dedup_countries = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[cols]

ax2_services = [
    'AFA','AMH','ARA','AZE','BEN','BUR','DAR','ECH','ELT','PER','FRE','GUJ','HAU','HIN','IGB','INO',
    'KOR','KRW','KYR','MAN','MAR','NEP','PAS','PDG','POL','POR','PUN','RUS','SER','SIN','SOM','SPA','SWA',
    'TAM','TEL','THA','TIG','TUR','UKR','URD','UZB','VIE','YOR', 'FOA', 'UKPS'
]

ax2_df_raw = weekly_df[weekly_df['ServiceID'].isin(ax2_services)].merge(country_codes, on='PlaceID', how='left')
ax2_df = pd.crosstab(
                    index = [ ax2_df_raw['PlaceID'], 
                              ax2_df_raw[pop_size_col], 
                              ax2_df_raw['w/c'],],
                    columns = ax2_df_raw['ServiceID'],
                    values =  ax2_df_raw['Reach'],
                    aggfunc='sum'
                ).reset_index()
ax2_df = ax2_df.fillna(0)

for col in ax2_services:
    if col not in ax2_df.columns:
        print(f'{col} missing!')
        ax2_df[col] = 0

temp2 = ax2_df.merge(africa_dedup_countries, on='PlaceID', how='left')
africa_df = temp2[~temp2['digiGAM_FOA_WT-'].isna()]
nonAfrica_df = temp2[temp2['digiGAM_FOA_WT-'].isna()]

# Apply the logic row-wise
def compute_value(row):
    others_sum = sum(row.get(code, 0) for code in ax2_services)
    if row['FOA'] > others_sum:
        return row['FOA'] + 0.60745497 * others_sum
    else:
        return others_sum + row['FOA'] * 0.60745497

africa_df['Reach'] = africa_df.apply(compute_value, axis=1)
# change to sum rather than
#nonAfrica_df = functions.sainsbury_formula(nonAfrica_df, gam_info['population_column'], ax2_services, 'Reach')
nonAfrica_df['Reach'] = nonAfrica_df[ax2_services].sum(axis=1)

ax2_df = pd.concat([africa_df, nonAfrica_df])

ax2_df['ServiceID'] = 'AX2'
ax2_df = ax2_df[['w/c', 'ServiceID', 'PlaceID', 'Reach']]

ax2_df.head()


# In[ ]:





# In[22]:


ax2_df_raw[
    (ax2_df_raw['PlaceID'] == 'POL') &
    #(digital_df['ServiceID'].isin(['AX2'])) &
    #(ax2_df_raw['PlatformID'].isin(['INS', 'WSC'])) 
    (ax2_df_raw['w/c'] == '2025-09-22')
]['Reach'].sum()


# ### ANW

# In[23]:


anw_services = ['AX2', 'WSE']
anw_df = compute_combined_reach(pd.concat([weekly_df, ax2_df]), anw_services, 'ANW', 
                                pop_size_col, country_codes)


# ### ANY

# In[24]:


any_services = ['ANW', 'GNL']
any_df = compute_combined_reach(pd.concat([weekly_df, anw_df]), any_services, 'ANY', 
                                pop_size_col, country_codes)


# ### TOT

# In[25]:


tot_services = ['ANY', 'MA-']
tot_df = compute_combined_reach(pd.concat([weekly_df, any_df]), tot_services, 'TOT', 
                                pop_size_col, country_codes, calc_type='add')


# ### ALL

# In[26]:


all_services = ['TOT', 'WOR']
all_df = compute_combined_reach(pd.concat([weekly_df, tot_df]), all_services, 'ALL', 
                                pop_size_col, country_codes, calc_type='add')


# ## finalising

# In[27]:


final_weekly_df = pd.concat([weekly_ws_df, weekly_ma_wor_df, 
                             enw_df, eng_df, en2_df, 
                             ax2_df, anw_df, any_df, tot_df, all_df])

final_weekly_df['PlatformID'] = platformID
final_weekly_df['YearGAE'] = gam_info['YearGAE']


# In[28]:


# SERVICE hierarchy issues
test_step = "calculated WSC reach"
service_hierarchy_issues = test_functions.test_hierarchy_reach(f"{platformID}_6_18", 
                                                               'Service', 
                                                               gam_info, 
                                                               final_weekly_df, 
                                                               ['w/c', 'PlaceID'],
                                                               metric_col='Reach',
                                                               test_step= test_step, 
                                                                round_metric=True)

# PLATFORM hierarchy issues
full_platform = pd.concat([singlePlatform_df, final_weekly_df])
full_platform['PlatformID'].unique()

test_step = "calculated WSC reach"
platform_hierarchy_issues = test_functions.test_hierarchy_reach(f"{platformID}_6_19", 
                                                               'Platform', 
                                                               gam_info, 
                                                               full_platform, 
                                                               ['w/c', 'PlaceID'],
                                                               metric_col='Reach',
                                                               test_step= test_step, 
                                                                round_metric=True)


# # store dataset

# In[29]:


'''ax2_ser = [
    'AFA','AMH','ARA','AZE','BEN','BUR','DAR','ECH','ELT','PER','FRE','GUJ','HAU','HIN','IGB','INO',
    'KOR','KRW','KYR','MAN','MAR','NEP','PAS','PDG','POR','PUN','RUS','SER','SIN','SOM','SPA','SWA',
    'TAM','TEL','THA','TIG','TUR','UKR','URD','UZB','VIE','YOR', 'FOA', 'UKPS'
]
final_weekly_df[
    (final_weekly_df['PlaceID'] == 'POL') &
    (final_weekly_df['ServiceID'].isin(['AX2'])) &
    (final_weekly_df['PlatformID'].isin(['INS', 'WSC'])) 
    & (final_weekly_df['w/c'] == '2025-09-22')
]
'''


# In[30]:


final_weekly_df.to_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_weekly_{platformID}.csv", 
                       index=None)


# In[31]:


final_annual_df = final_weekly_df.groupby(['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID'])['Reach'].sum().reset_index()
final_annual_df['Reach'] = final_annual_df['Reach'] / number_of_weeks

final_annual_df.to_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_annual_{platformID}.csv", 
                       index=None)


# In[ ]:





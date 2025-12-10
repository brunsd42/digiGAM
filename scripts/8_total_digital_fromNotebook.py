#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display 


import pandas as pd 
import missingno as msno
import numpy as np
from scipy.stats import zscore

import shutil
import os

from datetime import timedelta


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


platformID = 'CSS'


# In[4]:


# country
pop_size_col = gam_info['population_column']

country_cols = ['PlaceID', pop_size_col]
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)[country_cols]
country_codes[pop_size_col] = ( pd.to_numeric(country_codes[pop_size_col], errors='coerce')
                                    .fillna(1)
                                    .astype(int))

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period',)
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
today = pd.Timestamp.today()
last_monday = today - timedelta(days=today.weekday() + 7)
valid_weeks = week_tester[week_tester['w/c'] <= last_monday]
number_of_weeks = valid_weeks['WeekNumber_finYear'].max()

service_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='ServiceID',)

platform_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='PlatformID',)


### RUN Tests
test_functions.test_lookup_files(country_codes, country_cols, [f"{platformID}_0", f"{platformID}_1", f"{platformID}_2"], 
                                 test_step="lookup files - ensuring country codes is correct")

test_functions.test_lookup_files(week_tester, ['w/c'], [f"{platformID}_3", f"{platformID}_4", f"{platformID}_5"], 
                                 test_step = "lookup files - ensuring week tester is correct")

test_functions.test_lookup_files(service_tester, ['ServiceID'], [f"{platformID}_6", f"{platformID}_7", f"{platformID}_8"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")

test_functions.test_lookup_files(platform_tester, ['PlatformID'], [f"{platformID}_9", f"{platformID}_10", f"{platformID}_11"], 
                                 test_step = "lookup files - ensuring social media accounts is correct")


# In[5]:


# TODO Add tests (recognising all services, countries, platforms) to all the overlap sheets that are used here 
# overlap
overlap_nonHeavy = pd.read_excel("../data/stale/Final Overlaps 2021.xlsx", sheet_name='non heavy')
overlap_nonHeavy = overlap_nonHeavy.rename(columns={'Week': 'WeekNumber_finYear', 
                                                    'Service Code': 'ServiceID', 
                                                    'GeoCode': 'PlaceID'})
# ASK MINNIE WHY THE SHEET HAS MORE THAN ONE VALUE PER SERVICE/PLACE/WEEK
overlap_nonHeavy = overlap_nonHeavy.drop_duplicates(subset=['PlaceID', 'ServiceID', 'WeekNumber_finYear'], 
                                                    keep='first')
overlap_nonHeavy = overlap_nonHeavy.merge(week_tester[['w/c', 'WeekNumber_finYear']], 
                                          on='WeekNumber_finYear', 
                                          how='outer').drop(columns=['WeekNumber_finYear'])

overlap_nonHeavyAdd = pd.read_excel("../data/stale/Final Overlaps 2021.xlsx", sheet_name='non heav additional')
overlap_nonHeavyAdd = overlap_nonHeavyAdd.rename(columns={'GeoCode': 'PlaceID'})

overlap_SocWebOverlap = pd.read_excel("../data/stale/Final Overlaps 2021.xlsx", sheet_name='SocWebOverlap').drop_duplicates()
overlap_SocWebOverlap['PlaceID'] = overlap_SocWebOverlap['PlaceID'].replace('MYT', 'MAY').replace('WLF', 'WFI')
overlap_SocWebOverlap = overlap_SocWebOverlap.merge(country_codes, on='PlaceID', how='left')

overlap_referral = pd.read_excel("../data/stale/Final Overlaps 2021.xlsx", sheet_name='Referrals').drop_duplicates()
overlap_referral = overlap_referral.rename(columns={'Week Number': 'WeekNumber_finYear', 
                                                    'ServiceID': 'ServiceID', 
                                                    'Country Code': 'PlaceID',
                                                    '% Social': '%_AnalyticsSocialOverlap'})
overlap_referral = overlap_referral.merge(week_tester[['w/c', 'WeekNumber_finYear']], 
                                          on='WeekNumber_finYear', 
                                          how='outer').drop(columns=['WeekNumber_finYear'])

analytics_socialOverlap = 0.397690544

cols = ['PlaceID', 'digiGAM_FOA_WT-']
africa_dedup_countries = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[cols]


# # calculate CSS

# ## import data 

# ### site

# In[6]:


cols = ['YearGAE', 'w/c', 'ServiceID', 'PlatformID', 'PlaceID', 'Reach']
try:
    site_weekly_df = pd.read_csv(f"../data/singlePlatform/site/weekly/{gam_info['file_timeinfo']}_site_reach_weekly.csv")[cols]
    site_weekly_df['w/c'] = pd.to_datetime(site_weekly_df['w/c'])
    
    test_columns = {
    'PlaceID': country_codes['PlaceID'].tolist(),
    'w/c': week_tester['w/c'].tolist(),
    'ServiceID': service_tester['ServiceID'].tolist(),
    'PlatformID': platform_tester['PlatformID'].tolist()
    }
    
    for i, (column, allowed_values) in enumerate(test_columns.items(), start=1):
        label = f"total_digi_{i}"
        test_functions.test_allowed_values(site_weekly_df, column, allowed_values, label, 'site_ingest')
# test duplicates in sites
except:
    site_weekly_df = pd.DataFrame()
    print('site not there!')


# ### social

# In[7]:


social_weekly_df = pd.read_csv("../data/combinePlatforms/GAM2026_weekly_WSC.csv")

social_weekly_df['YearGAE'] = gam_info['YearGAE']
social_weekly_df = social_weekly_df[['ServiceID', 'PlaceID', 'PlatformID', 'w/c', 'YearGAE', 'Reach']]


test_columns = {
    'PlaceID': country_codes['PlaceID'].tolist(),
    'w/c': week_tester['w/c'].tolist(),
    'ServiceID': service_tester['ServiceID'].tolist(),
    'PlatformID': platform_tester['PlatformID'].tolist()
}

for i, (column, allowed_values) in enumerate(test_columns.items(), start=5):
    print(column)
    label = f"total_digi_{i}"
    test_functions.test_allowed_values(social_weekly_df, column, allowed_values, label, 'social_ingest')
social_weekly_df.head()


# ### combine site & social

# In[8]:


site_social_df = pd.concat([site_weekly_df, social_weekly_df], ignore_index=True)
site_social_df['w/c'] = pd.to_datetime(site_social_df['w/c'])
site_social_df = site_social_df.pivot(index=['YearGAE', 'w/c', 'ServiceID', 'PlaceID'],
                                      columns="PlatformID", values="Reach").fillna(0).reset_index()
site_social_df.sample(5)


# In[9]:


non_MA_WOR = site_social_df[~site_social_df['ServiceID'].isin(['WOR', 'MA-'])]

exclude_ids = ['WOR', 'MA-', 
               'ENG', 'EN2', 'ENW', 
               'ANW', 'TOT', 'AX2', 'ANY', 'ALL', ]
ws_site_social = site_social_df[~site_social_df['ServiceID'].isin(exclude_ids)]


ma_wor_df = site_social_df[site_social_df['ServiceID'].isin(['WOR', 'MA-'])]


# ## WSL

# ### determine & handle outlier

# In[10]:


# Ensure required columns exist
for col in ['WSC', 'WDI']: 
    if col not in ws_site_social.columns:
        ws_site_social[col] = 0  # Fill missing column with zeros

# Group by 'service' and 'w/c', summing the numerical columns
grouped = ws_site_social.groupby(['ServiceID', 'w/c'], as_index=False)[['WSC', 'WDI']].sum()

# Compute Z-scores within each 'service' group
def compute_zscores(group):
    group['WSC_z'] = (group['WSC'] - group['WSC'].mean()) / group['WSC'].std(ddof=1)
    group['WDI_z'] = (group['WDI'] - group['WDI'].mean()) / group['WDI'].std(ddof=1)
    return group

outlier_df = grouped.groupby('ServiceID', group_keys=False).apply(compute_zscores)
outlier_df = outlier_df[(outlier_df['WSC_z'] > 1.96) | (outlier_df['WDI_z'] > 1.96)]
outlier_df = outlier_df[['w/c', 'ServiceID']]


# In[11]:


# 2. Identify outliers and non-outliers
merged_outlier = ws_site_social.merge(outlier_df, on=['w/c', 'ServiceID'], how='outer', 
                                   indicator=True)


# In[12]:


outliers = merged_outlier[merged_outlier['_merge'] == 'both']
print(outliers.shape)

# 3. Check for overlap with non-heavy overlaps
outliers = outliers.drop(columns=['_merge'])
outliers = outliers.merge(overlap_nonHeavy.rename(columns={'Non Heavy - Month': 'NonHeavy1'}), 
                                 on=['ServiceID', 'PlaceID', 'w/c'], 
                                 how='left')
print(outliers.shape)

# 4. Add additional overlap info
outliers = outliers.merge(overlap_nonHeavyAdd.rename(columns={'Avg_%': 'NonHeavy2'}), 
                                        on='PlaceID', 
                                        how='left')
print(outliers.shape)

outliers['NonHeavy3'] = 0.593887
outliers['%_NonHeavy'] = np.where(outliers['NonHeavy1'].notna(), 
                                         outliers['NonHeavy1'],
                                         np.where(outliers['NonHeavy2'].notna(), 
                                                  outliers['NonHeavy2'], 
                                                  outliers['NonHeavy3']))
outliers = outliers.drop(columns=['NonHeavy1', 'NonHeavy2', 'NonHeavy3'])

outliers['NonHeavyWDI'] = outliers['WDI'] * outliers['%_NonHeavy']
outliers['HeavyWDI'] = outliers['WDI'] * (1-outliers['%_NonHeavy'])

outliers['NonHeavyWSC'] = outliers['WSC'] * outliers['%_NonHeavy']
outliers['HeavyWSC'] = outliers['WSC'] * (1-outliers['%_NonHeavy'])

# be aware - PCN is missing 
outliers = outliers.merge(overlap_SocWebOverlap, on='PlaceID', how='left')
print(outliers.shape)

# Apply to your DataFrame
outliers = functions.sainsbury_formula(outliers, 'Population 2020', ['NonHeavyWDI', 'NonHeavyWSC'], 'NonHeavy_WSC_WDI')

# BUFFER
def compute_heavy_combined(row):
    heavy_wdi = row['HeavyWDI']
    heavy_wsc = row['HeavyWSC']
    social_inc = row['Social Incremental']
    site_inc = row['Web']
    
    if heavy_wdi > heavy_wsc:
        return heavy_wdi + heavy_wsc * social_inc
    else:
        return heavy_wsc + heavy_wdi * site_inc

# Apply to your DataFrame
outliers['BUFFER_Heavy_WSC_WDI'] = outliers.apply(compute_heavy_combined, axis=1)
outliers['Buffer_WSC_WDI'] = outliers['NonHeavy_WSC_WDI'] + outliers['BUFFER_Heavy_WSC_WDI']
outliers['Heavy_WSC_WDI'] = (outliers['HeavyWSC'] + outliers['HeavyWDI']) * outliers['Own Web & Social Factor']
outliers['WSC_WDI'] = outliers['Heavy_WSC_WDI'] + outliers['NonHeavy_WSC_WDI']

def compute_wsc_wdi_buffer(row):
    wsc_wdi = row['WSC_WDI']
    wdi = row['WDI']
    wsc = row['WSC']
    buffer = row['Buffer_WSC_WDI']
    
    if wsc_wdi < wdi:
        return buffer
    elif wsc_wdi < wsc:
        return buffer
    else:
        return wsc_wdi

# Ensure required columns exist
for col in ['WIN', 'WWW']: 
    if col not in outliers.columns:
        outliers[col] = 0  # Fill missing column with zeros

outliers['WSC_WDI'] = outliers.apply(compute_wsc_wdi_buffer, axis=1)
outliers = outliers[['PlaceID', 'ServiceID', 'YearGAE', 'w/c', 
                     'WDI', 'WIN', 'WSC', 'WWW', 
                     'WSC_WDI']]


# ### handle non outlier

# In[13]:


no_outliers = merged_outlier[merged_outlier['_merge'] == 'left_only']
overlap_cols = ['PlaceID', pop_size_col, 'Own Web & Social Factor', 'Web', 'Social Incremental']
no_outliers = no_outliers.merge(overlap_SocWebOverlap[overlap_cols], on='PlaceID', how='left')


def compute_web_social_value(row):
    wdi = row['WDI']
    wsc = row['WSC']
    factor = row['Own Web & Social Factor']
    
    if wsc == 0:
        return wdi
    elif wdi == 0:
        return wsc
    else:
        return (wdi + wsc) * factor

# Apply to your DataFrame
no_outliers['NewTapestryDigi'] = no_outliers.apply(compute_web_social_value, axis=1)

def compute_web_social_mix(row):
    wdi = row['WDI']
    wsc = row['WSC']
    web = row['Web']
    social_inc = row['Social Incremental']
    
    if wsc > wdi:
        return wsc + wdi * web
    else:
        return wdi + wsc * social_inc

# Apply to your DataFrame
no_outliers['BufferForTapestry'] = no_outliers.apply(compute_web_social_mix, axis=1)


def compute_tapestry_buffer(row):
    digi = row['NewTapestryDigi']
    wsc = row['WSC']
    wdi = row['WDI']
    buffer = row['BufferForTapestry']
    
    if digi < wsc:
        return buffer
    elif digi < wdi:
        return buffer
    else:
        return digi

# Apply to your DataFrame
# Ensure required columns exist
for col in ['WIN', 'WWW']: 
    if col not in no_outliers.columns:
        no_outliers[col] = 0  # Fill missing column with zeros

no_outliers['WSC_WDI'] = no_outliers.apply(compute_tapestry_buffer, axis=1)
no_outliers = no_outliers[['PlaceID', 'ServiceID', 'YearGAE', 'w/c',
                           'WDI', 'WIN', 'WSC', 'WWW',
                           'WSC_WDI']]


# ### calculate CSS

# In[45]:


ws_site_social_postOutlier = pd.concat([outliers, no_outliers])
ws_site_social_postOutlier['%_OverlapWithOwnSite'] = ((ws_site_social_postOutlier['WSC'] +
                                                     ws_site_social_postOutlier['WDI']) -
                                                    ws_site_social_postOutlier['WSC_WDI'])/ws_site_social_postOutlier['WDI']

ws_site_social_postOutlier = ws_site_social_postOutlier.merge(overlap_referral, 
                                                              on=['PlaceID', 'ServiceID', 'w/c'],
                                                              how='left', indicator=True)

ws_site_social_postOutlier['%_AnalyticsSocialOverlap'] = ws_site_social_postOutlier['%_AnalyticsSocialOverlap'].fillna(analytics_socialOverlap)


def compute_site_overlap_adjustment(row):
    site_overlap = row['%_OverlapWithOwnSite']
    analytics_overlap = row['%_AnalyticsSocialOverlap']
    wdi = row['WDI']
    wsc = row['WSC']
    wsc_wdi = row['WSC_WDI']
    
    if site_overlap < analytics_overlap:
        return (wdi + wsc) - (wdi * analytics_overlap)
    else:
        return wsc_wdi
ws_site_social_postOutlier['Pegged_WSC_WDI'] = ws_site_social_postOutlier.apply(compute_site_overlap_adjustment, axis=1)

def flag_pegged_wsc_wdi(row):
    pegged = row['Pegged_WSC_WDI']
    wsc = row['WSC']
    wdi = row['WDI']
    
    if pegged < wsc:
        return "FLAG"
    elif pegged < wdi:
        return "FLAG"
    else:
        return "FINE"
ws_site_social_postOutlier['Check1'] = ws_site_social_postOutlier.apply(flag_pegged_wsc_wdi, axis=1)
print(ws_site_social_postOutlier['Check1'].value_counts())

def resolve_check_flag(row):
    check = row['Check1']
    wsc_wdi = row['WSC_WDI']
    pegged = row['Pegged_WSC_WDI']
    
    if check == "FLAG":
        return wsc_wdi
    else:
        return pegged

# Apply to your DataFrame
ws_site_social_postOutlier['final_WSC_WDI'] = ws_site_social_postOutlier.apply(resolve_check_flag, axis=1)

def compute_incremental_partner_reach(row):
    www = row['WWW']
    win = row['WIN']
    wdi = row['WDI']
    final_wsc_wdi = row['final_WSC_WDI']
    
    if www > win:
        return (www - wdi) + final_wsc_wdi
    elif win == 0:
        return final_wsc_wdi
    elif final_wsc_wdi == 0:
        return win
    else:
        return (www - wdi) + final_wsc_wdi

# Apply to your DataFrame
ws_site_social_postOutlier['Reach'] = ws_site_social_postOutlier.apply(compute_incremental_partner_reach, axis=1)


# In[15]:


cols = ['YearGAE', 'w/c', 'ServiceID', 'PlaceID', 'Reach']
weekly_ws_df =  ws_site_social_postOutlier[cols]


# ### annual average

# In[16]:


annual_ws_df = weekly_ws_df.groupby(['YearGAE', 'ServiceID', 'PlaceID'])['Reach'].sum().reset_index()
annual_ws_df['Reach'] = annual_ws_df['Reach'] / number_of_weeks
annual_ws_df.head()


# ## MA & Studios

# In[17]:


ma_wor_df = site_social_df[site_social_df['ServiceID'].isin(['WOR', 'MA-'])]
ma_wor_df = ma_wor_df.merge(overlap_SocWebOverlap, on='PlaceID' , how='left')

# Ensure required columns exist
for col in ['WSC', 'WWW']: 
    if col not in ma_wor_df.columns:
        ma_wor_df[col] = 0  # Fill missing column with zeros

ma_wor_df = functions.sainsbury_formula(ma_wor_df, pop_size_col, ['WSC', 'WWW'], 'Reach')

cols = ['YearGAE', 'w/c', 'ServiceID', 'PlaceID', 'Reach']
weekly_ma_wor_df = ma_wor_df[cols]

annual_ma_wor_df = ma_wor_df.groupby(['YearGAE', 'ServiceID', 'PlaceID'])['Reach'].sum().reset_index()
annual_ma_wor_df['Reach'] = annual_ma_wor_df['Reach'] / number_of_weeks


# ## aggregated services

# In[18]:


weekly_df = pd.concat([weekly_ws_df, weekly_ma_wor_df])
#weekly_df = weekly_df.merge(country_codes, on='PlaceID', how='left')
weekly_df.head()


# In[19]:


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
               filtered_df['w/c'], filtered_df['YearGAE']],
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
    return pivot_df[['YearGAE', 'w/c', 'ServiceID', 'PlaceID', 'Reach']]


# ### ENW

# In[20]:


# Usage
enw_services = ['FOA', 'WSE']
enw_df = compute_combined_reach(weekly_df, enw_services, 'ENW', pop_size_col, country_codes)


# ### ENG

# In[21]:


# Usage
eng_services = ['GNL', 'WSE']
eng_df = compute_combined_reach(weekly_df, eng_services, 'ENG', pop_size_col, country_codes)


# ### EN2

# In[22]:


en2_services = ['ENG', 'WOR']
en2_df = compute_combined_reach(pd.concat([weekly_df, eng_df]), en2_services, 'EN2', pop_size_col, country_codes)


# ### AX2

# In[23]:


ax2_services = [
    'AFA','AMH','ARA','AZE','BEN','BUR','DAR',#'ECH',
    'ELT','PER','FRE','GUJ','HAU','HIN','IGB','INO',
    'KOR','KRW','KYR','MAN','MAR','NEP','PAS','PDG','POR', 'POL', 'PUN','RUS','SER','SIN','SOM','SPA','SWA',
    'TAM','TEL','THA','TIG','TUR','UKR','URD','UZB','VIE','YOR', 'FOA', #'UKPS'
]

ax2_df = weekly_df[weekly_df['ServiceID'].isin(ax2_services)].merge(country_codes, on='PlaceID', how='left')
ax2_df = pd.crosstab(
                                        index = [ ax2_df['PlaceID'], 
                                                  ax2_df[pop_size_col], 
                                                  ax2_df['w/c'],
                                                  ax2_df['YearGAE']],
                                        columns = ax2_df['ServiceID'],
                                        values =  ax2_df['Reach'],
                                        aggfunc='sum'
                                    ).reset_index()
ax2_df = ax2_df.fillna(0)

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
nonAfrica_df = functions.sainsbury_formula(nonAfrica_df, gam_info['population_column'], ax2_services, 'Reach')
ax2_df = pd.concat([africa_df, nonAfrica_df])
ax2_df['ServiceID'] = 'AX2'
ax2_df = ax2_df[['YearGAE', 'w/c', 'ServiceID', 'PlaceID', 'Reach']]

ax2_df.head()


# ### ANW

# In[24]:


anw_services = ['AX2', 'WSE']
anw_df = compute_combined_reach(pd.concat([weekly_df, ax2_df]), anw_services, 'ANW', 
                                pop_size_col, country_codes)


# ### ANY

# In[25]:


any_services = ['ANW', 'GNL']
any_df = compute_combined_reach(pd.concat([weekly_df, anw_df]), any_services, 'ANY', 
                                pop_size_col, country_codes)


# ### TOT

# In[26]:


tot_services = ['ANY', 'MA-']
tot_df = compute_combined_reach(pd.concat([weekly_df, any_df]), tot_services, 'TOT', 
                                pop_size_col, country_codes, calc_type='add')


# ### ALL

# In[27]:


all_services = ['TOT', 'WOR']
all_df = compute_combined_reach(pd.concat([weekly_df, tot_df]), all_services, 'ALL', 
                                pop_size_col, country_codes, calc_type='add')


# ## finalising 

# In[28]:


final_weekly_df = pd.concat([weekly_ws_df, weekly_ma_wor_df, enw_df, eng_df, en2_df, ax2_df, anw_df,
                     any_df, tot_df, all_df])

final_weekly_df['PlatformID'] = 'CSS'

final_weekly_df.to_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_weekly_CSS.csv", 
                       index=None)


# In[29]:


final_annual_df = final_weekly_df.groupby(['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID'])['Reach'].sum().reset_index()
final_annual_df['Reach'] = final_annual_df['Reach'] / number_of_weeks

final_annual_df.to_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_annual_CSS.csv", 
                       index=None)


# # combine platforms
# 
# ## import data

# In[30]:


cols = ['YearGAE', 'w/c', 'ServiceID', 'PlatformID', 'PlaceID', 'Reach']


# In[31]:


podcast_df = pd.read_excel(f"../data/singlePlatform/podcast/weekly/{gam_info['file_timeinfo']}_podcast_data_weekly.xlsx")
podcast_df['YearGAE'] = gam_info['YearGAE']
podcast_df['w/c'] = podcast_df['w/c'].dt.strftime('%Y-%m-%d')
podcast_df = podcast_df[cols]


# In[32]:


try:
    site_df = pd.read_csv(f"../data/singlePlatform/site/weekly/{gam_info['file_timeinfo']}_site_reach_weekly.csv", index_col=0)
    site_df = site_df[cols]
    site_df.head()
except:
    print('site not there! ')
    site_df = pd.DataFrame()


# In[33]:


# created from platform in 6.
social_wsc_df = pd.read_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_weekly_WSC.csv")
social_wsc_df = social_wsc_df.rename(columns={
    'Country Code': 'PlaceID',
    'Platform Code': 'PlatformID',
    'Service Code': 'ServiceID'
})
social_wsc_df['YearGAE'] = gam_info['YearGAE']
social_wsc_df = social_wsc_df[cols]
social_wsc_df.head()


# In[34]:


# created from dataset per platform file in 5.
social_platforms_df = pd.read_csv(f"../data/combinePlatforms/social_media_data_{gam_info['file_timeinfo']}_platform_weekly.csv")
social_platforms_df['YearGAE'] = gam_info['YearGAE']
social_platforms_df = social_platforms_df[cols]
social_platforms_df.head()


# In[35]:


social_platforms_df['PlatformID'].unique()


# In[36]:


css_df = pd.read_csv(f"../data/combinePlatforms/{gam_info['file_timeinfo']}_weekly_CSS.csv")[cols]
css_df.head()


# # combine 

# In[37]:


sources = {'pod': podcast_df, 
           'site': site_df, 
           'platform': social_platforms_df, 
           'wsc': social_wsc_df,
           'css': css_df}

def far_per_test(df):
    temp = df[df['ServiceID'].isin(['PER', 'FAR'])]
    display(temp.ServiceID.value_counts())
    if len(temp) > 0:
        df['ServiceID'] = df['ServiceID'].replace('FAR', 'PER')
    temp = df[df['ServiceID'].isin(['PER', 'FAR'])]
    display(temp.ServiceID.value_counts())
    return df

for name, source in sources.items():
    if len(source) > 0:
        print(f"\n{name}")
        sources[name] = far_per_test(source)
    else:
        print(f"\n{name} is empty! ")
digital_df = pd.concat(sources.values())

# Paths for final outputs
output_dir = "../data/final"
os.makedirs(output_dir, exist_ok=True)

digital_df = digital_df[digital_df['ServiceID'] != 'AXE']
digital_df.to_csv(f"{output_dir}/{gam_info['file_timeinfo']}_digi_gam_weekly.csv", 
                       index=None)

digital_annual_df = digital_df.groupby(['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID'])['Reach'].sum().reset_index()
digital_annual_df['Reach'] = digital_annual_df['Reach'] / number_of_weeks

digital_annual_df.to_csv(f"{output_dir}/{gam_info['file_timeinfo']}_digi_gam_annual.csv", 
                       index=None)

# ✅ Copy the test logbook into the same folder
logbook_src = "../test/test_logbook.xlsx"
logbook_dest = f"{output_dir}/{gam_info['file_timeinfo']}_test_logbook.xlsx"

if os.path.exists(logbook_src):
    shutil.copy(logbook_src, logbook_dest)
    print(f"Test logbook copied to: {logbook_dest}")
else:
    print("Warning: Test logbook not found!")


# In[38]:


digital_df['PlatformID'].unique()


# In[39]:


digital_df['w/c'].unique()


# In[44]:


len(digital_df['w/c'].unique())


# In[42]:


ax2_ser = [
    'AFA','AMH','ARA','AZE','BEN','BUR','DAR','ECH','ELT','PER','FRE','GUJ','HAU','HIN','IGB','INO',
    'KOR','KRW','KYR','MAN','MAR','NEP','PAS','PDG','POR','PUN','RUS','SER','SIN','SOM','SPA','SWA',
    'TAM','TEL','THA','TIG','TUR','UKR','URD','UZB','VIE','YOR', 'FOA', 'UKPS'
]
digital_df[
    (digital_df['PlatformID'].isin(['YT-', 'WSC'])) 
    & (digital_df['w/c'] == '2025-11-24')
]


# In[41]:


#SWA 
#2025-


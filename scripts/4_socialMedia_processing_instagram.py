#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tqdm import tqdm 
from datetime import datetime

import pandas as pd
pd.set_option('display.float_format', '{:.00f}'.format)

import numpy as np

import missingno as msno


# In[4]:


platformID = 'INS'


# ## import helper

# In[5]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

import test_functions 
import functions 


# In[6]:


# country
pop_size_col = gam_info['population_column']
cols = ['PlaceID', pop_size_col]
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')[cols]

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                            sheet_name='GAM Period',)
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts
socialmedia_accounts = pd.read_excel("helper/ins_account_lookup.xlsx")
socialmedia_accounts = socialmedia_accounts.rename(columns={'IG studios exc uk': 'Excluding UK'})
# overlaps 
overlaps = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='overlap')
overlaps.head()


# ## import data 

# In[7]:


full_df = pd.read_csv(f"../data/processed/{platformID}/{gam_info['file_timeinfo']}_uniqueViewer_country.csv")
# rename / replace to the appropriate columns 
'''
- PlatformID
- YearGAE
- ServiceID
- Week Number
- Reach
'''
print(full_df.shape)

full_df = full_df.merge(country_codes, on='PlaceID', 
                        how='left', indicator=True)
print(full_df._merge.value_counts())
full_df.drop(columns=['_merge'], inplace=True)
full_df.sample()
full_df.head()


# ## functions

# In[8]:


'''def process_overlap(data, service1, service2, grouped_service, overlap_type, 
                    overlap_service_id, platformID, gam_info, path):
    # Ensure the grouped_service key exists
    if grouped_service not in data:
        data[grouped_service] = {}

    # Extract weekly data
    df1 = data[service1]['weekly']
    df2 = data[service2]['weekly']
    
    # Concatenate
    combined_df = pd.concat([df1, df2])
    
    # Pivot
    pivot_df = pd.crosstab(
        index=[combined_df['PlaceID'], combined_df['w/c']],
        columns=combined_df['ServiceID'],
        values=combined_df['Reach'],
        aggfunc='sum'
    ).reset_index()
    
    # Fill missing values
    pivot_df[service1] = pivot_df[service1].fillna(0)
    pivot_df[service2] = pivot_df[service2].fillna(0)
    
    # Get overlap
    if overlap_type != 'sainsbury':
        overlap_df = overlaps[overlaps['Overlap Type'] == overlap_type]
        overlap_value = overlap_df.loc[overlap_df['ServiceID'] == overlap_service_id, 'overlap_%'].values[0]
        print(f"overlap applied: {overlap_value}")
        pivot_df['overlap'] = overlap_value
        
        # Calculate adjusted reach
        pivot_df['Reach'] = pivot_df.apply(
            lambda row: row[service1] + row[service2] * (1 - row['overlap']) 
            if row[service1] > row[service2] 
            else row[service1] * (1 - row['overlap']) + row[service2],
            axis=1
        )
    else: 
        # add population
        pivot_df = pivot_df.merge(country_codes, on='PlaceID', how='left', indicator=True)
        print(f"adding population: {pivot_df._merge.value_counts()}")
        pivot_df = pivot_df.drop(columns=['_merge'])
        
        services = [service1, service2]
        pivot_df = functions.sainsbury_formula(pivot_df, pop_size_col, 
                                      services, 'Reach')
        
    # Assign grouped service
    pivot_df['ServiceID'] = grouped_service
    
    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)
    
    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = functions.calculate_weekly_sumServices(pivot_df, grouped_service, platformID, gam_info)
    annual_df = functions.calculate_annualy(data[grouped_service]['weekly'], platformID, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df
    
    return pivot_df, annual_df
'''


# In[9]:


'''def process_overlap_v2(data, service1, service2, grouped_service,
                    overlap_type, overlap_service_id, platformID, gam_info, path,
                    service3=None):  # <-- Add service3 as an optional argument
    """
    overlap_service_id = which service ID contains the overlap factor!
    service3: only used for overlap_type 'sainsbury'
    """
    # Ensure the grouped_service key exists
    if grouped_service not in data:
        data[grouped_service] = {}

    # Extract weekly data
    df1 = data[service1]['weekly']
    df2 = data[service2]['weekly']
    
    # For sainsbury, include service3
    if service3 is not None:
        df3 = data[service3]['weekly']
        combined_df = pd.concat([df1, df2, df3])
        services = [service1, service2, service3]
    else:
        combined_df = pd.concat([df1, df2])
        services = [service1, service2]
    
    # Pivot
    pivot_df = pd.crosstab(
        index=[combined_df['PlaceID'], combined_df['w/c']],
        columns=combined_df['ServiceID'],
        values=combined_df['Reach'],
        aggfunc='sum'
    ).reset_index()
    
    # Fill missing values for all services
    for service in services:
        if service in pivot_df.columns:
            pivot_df[service] = pivot_df[service].fillna(0)
    
    # Get overlap
    if overlap_type != 'sainsbury':
        if grouped_service == 'EN2':
            
            pivot_df['Reach'] = np.where(
                        (pivot_df['GNL'] + pivot_df['WSE']) > pivot_df['WOR'],
                        (pivot_df['GNL'] + pivot_df['WSE']) + (0.892857142857143 * pivot_df['WOR']),
                        pivot_df['WOR'] + ((pivot_df['GNL'] + pivot_df['WSE']) * 0.952380952380952)
                    )

        else:
            overlap_df = overlaps[overlaps['Overlap Type'] == overlap_type]
            overlap_value = overlap_df.loc[overlap_df['ServiceID'] == overlap_service_id, 'overlap_%'].values[0]
            print(f"overlap applied: {overlap_value}")
            pivot_df['overlap'] = overlap_value
            
            # Calculate adjusted reach (unchanged)
            pivot_df['Reach'] = pivot_df.apply(
                lambda row: row[service1] + row[service2] * (1 - row['overlap']) 
                if row[service1] > row[service2] 
                else row[service1] * (1 - row['overlap']) + row[service2],
                axis=1
            )
        
    else: 
        # add population
        pivot_df = pivot_df.merge(country_codes, on='PlaceID', how='left', indicator=True)
        print(f"adding population: {pivot_df._merge.value_counts()}")
        pivot_df = pivot_df.drop(columns=['_merge'])
        
        # Pass all services to sainsbury_formula
        pivot_df = functions.sainsbury_formula(pivot_df, pop_size_col, services, 'Reach')
            
    # Assign grouped service
    pivot_df['ServiceID'] = grouped_service
    
    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)
    
    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = functions.calculate_weekly_sumServices(pivot_df, grouped_service, platformID, gam_info)
    annual_df = functions.calculate_annualy(data[grouped_service]['weekly'], platformID, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df
    
    return pivot_df, annual_df
'''


# # calculate 

# In[10]:


path = f"../data/singlePlatform/{platformID}/"


# ## Business Units

# In[11]:


data = {}
'''temp_bus = ['GNL', 'WSL', 'GNL','Studios', 'WSE', 'MA-', 'FOA']
for bu in temp_bus:
'''#
for bu in gam_info['business_units'].keys():
    print(f"### processing {bu} ######################################################")
    data[bu] = {'weekly': 'tbd', 
                #'annual': 'tbd'
               }
    
    bu_configs = gam_info['business_units'][bu]
    print(bu_configs)
    df = full_df[full_df['ServiceID'].isin(gam_info['business_units'][bu]['Service IDs'])]
    
    if df.empty:
        print(f"no data yet for {bu}")
        
    channel_ids = df['Channel ID'].unique().tolist()
    
    # will include / exclude the uk based on bu_configs
    df = functions.include_uk_decision(df, socialmedia_accounts)
    
    # for later testing or if sainsbury isn't used 
    summed_uv_by_country = df.groupby(['ServiceID', 'w/c', 'PlaceID'])\
                                .agg({'uv_by_country': 'sum'})\
                                .reset_index()
    
    if bu_configs['sainsbury'][platformID]:
        print('sainsbury is applied')
        # pivot 
        channel_uv_by_country = pd.crosstab(
                                        index = [ df['PlaceID'], 
                                                  df['ServiceID'], 
                                                  df[pop_size_col], 
                                                  df['w/c']],
                                        columns = df['Channel ID'],
                                        values =  df['uv_by_country'],
                                        aggfunc='sum'
                                    ).reset_index()
    
        # check for missing values
        # especially in the string columns no values should be missing
        msno.matrix(channel_uv_by_country)
        
        # fill missing values with 0 - this is good fi the matrix above showed that the string 
        # columns did not have any missings so the only gaps filled are numeric. 
        channel_uv_by_country = channel_uv_by_country.fillna(0)
        
        #calculate sainsbury
        channel_uv_by_country = functions.sainsbury_formula(channel_uv_by_country, pop_size_col, 
                                      channel_ids, 'uv_by_country')
        
        cols_left =  ['w/c', 'PlaceID', 'uv_by_country']
        cols_right = ['w/c', 'PlaceID', 'ServiceID', 'uv_by_country']
        #yt_deduped = channel_uv_by_country[cols_left].merge(summed_uv_by_country[cols_right], on=['w/c', 'PlaceID'], how='inner')
        yt_deduped = channel_uv_by_country.rename(columns={'uv_by_country': 'Reach'})
        
    else:
        print('sainsbury is skipped ')
        # instead of pivot we can use the summed df above that already contains the sum over 
        # YT Service Code so the channels are already summarised in Services
        yt_deduped = summed_uv_by_country.rename(columns={'uv_by_country': 'Reach'})
    
    # processing 
    weekly_df= functions.summary_excel(yt_deduped, bu, platformID, gam_info)
    
    # storing data
    data[bu]['weekly'] = weekly_df
    #data[bu]['annual'] = annual_df
    
    


# ## AXE

# In[12]:


grouped_service = 'AXE'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}
    


# ### weekly 

# In[13]:


wsl_weekly = data['WSL']['weekly']
wsl_weekly = wsl_weekly[~wsl_weekly['ServiceID'].isin(['SER', 'SIN'])]

data[grouped_service]['weekly'] = functions.calculate_weekly_sumServices(wsl_weekly, grouped_service, platformID, gam_info)


# ### annualy

# In[14]:


'''axe_annual = functions.calculate_annualy(data[grouped_service]['weekly'], platformID, gam_info)#, aggregation_type='old')
path = "../data/singlePlatform/"
file_path = f"{gam_info['file_timeinfo']}_{platform}_{grouped_service}.xlsx"
axe_annual.to_excel(path+file_path, index=None)
data[grouped_service]['annual'] = axe_annual
file_path'''


# ## AX2
# (WSL + Africa)
# 

# In[15]:


grouped_service = 'AX2'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}
    


# In[16]:


pivot_ax, annual_ax = functions.process_overlap(
    data=data,
    service1='FOA',
    service2='AXE',
    grouped_service='AX2',
    overlaps=overlaps,
    overlap_type='WSL/FOA',
    overlap_service_id='FOA',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)



# ## ANW

# In[17]:


grouped_service = 'ANW'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}
    
pivot_anw, annual_anw = functions.process_overlap(
    data=data,
    service1='AX2',
    service2='WSE',
    grouped_service='ANW',
    overlaps=overlaps,
    overlap_type='WSE/WSL',
    overlap_service_id='AXE',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# ## ANY 
# WS + GN

# In[18]:


grouped_service = 'ANY'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap(
    data=data,
    service1='GNL',
    service2='ANW',
    grouped_service='ANY',
    overlaps=overlaps,
    overlap_type='WSL/GNL',
    overlap_service_id='ANW',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# ## TOT 
# WS GNL MA

# In[19]:


grouped_service = 'TOT'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap(
    data=data,
    service1='MA-',
    service2='ANY',
    grouped_service='TOT',
    overlap_type='sainsbury',
    overlap_service_id='-',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# ## ALL
# TOT + WOR

# In[20]:


grouped_service = 'ALL'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap(
    data=data,
    service1='TOT',
    service2='WOR',
    grouped_service='ALL',
    overlap_type='sainsbury',
    overlap_service_id='-',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# ## ENG

# In[21]:


grouped_service = 'ENG'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap(
    data=data,
    service1='GNL',
    service2='WSE',
    grouped_service='ENG',
    overlap_type='sainsbury',
    overlap_service_id='-',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# ## EN2 

# In[22]:


grouped_service = 'EN2'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap_v2(
    data=data,
    service1='GNL',
    service2='WSE',
    grouped_service=grouped_service,
    overlaps=overlaps,
    overlap_type='other',
    overlap_service_id='-',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col,
    service3='WOR'
)


# ## ENW

# In[23]:


grouped_service = 'ENW'
data[grouped_service] = {'weekly': 'tbd',
                         'annual': 'tbd'}

pivot_any, annual_any = functions.process_overlap(
    data=data,
    service1='WSE',
    service2='FOA',
    grouped_service=grouped_service,
    overlap_type='sainsbury',
    overlap_service_id='-',
    platformID=platformID,
    gam_info=gam_info,
    path=path,
    country_codes=country_codes, 
    pop_size_col=pop_size_col
)


# # Consolidation

# In[24]:


'''consolidated_dfs = []
for service in data.keys():
    consolidated_dfs.append(data[service]['annual'])
consolidated_df = pd.concat(consolidated_dfs)

totals = consolidated_df[consolidated_df['PlaceID'] == 'Total']
non_totals = consolidated_df[consolidated_df['PlaceID'] != 'Total']'''


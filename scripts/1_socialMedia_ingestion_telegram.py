#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install pyrogram')


# In[2]:


import pandas as pd 
pd.set_option('display.float_format', '{:.00f}'.format)

from pyrogram import Client
import asyncio
from tqdm import tqdm
import nest_asyncio
from datetime import datetime
import os

import numpy as np


# In[3]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config import gam_info
from security_config import telegram_key

from functions import execute_sql_query
import test_functions


# In[4]:


platformID = 'TEL'
# service
service_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='ServiceID')

# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID', 
                              keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts_all = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')
socialmedia_accounts = socialmedia_accounts_all[socialmedia_accounts_all['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']


# # get data

# In[5]:


nest_asyncio.apply()


# In[6]:


async def fetch_and_merge_data(channel_name, socialmedia_accounts, platformID):
    
    async with Client("my_account", telegram_key['api_id'], telegram_key['api_hash']) as app:
        message_data = []
        async for message in app.get_chat_history(channel_name):
            if message.date > datetime(2024, 3, 1):
                message_data.append(message.__dict__)
            else:
                break  # Stop fetching older messages

    new_df = pd.DataFrame(message_data)

    file_name = f"{socialmedia_accounts['ServiceID'].value[0]}_{platformID}.csv"
    new_df.to_csv(csv_file, index=False)
    return new_df

channel_names = socialmedia_accounts['Channel ID'].tolist()
tel_raw = []
for channel_name in tqdm(channel_names):
    result = await fetch_and_merge_data(channel_name, socialmedia_accounts, platformID)
    tel_raw.append(result)


# In[7]:


from pyrogram import Client

app = Client("my_account", api_id=telegram_key["api_id"], api_hash=telegram_key["api_hash"])
print(app.session_name)
print(app.workdir)


# In[8]:


import os

# Check for session file in current working directory
session_filename = "my_account.session"
session_path = os.path.join(os.getcwd(), session_filename)

if os.path.exists(session_path):
    print(f"✅ Session file exists at: {session_path}")
else:
    print(f"❌ No session file found at: {session_path}")


# # calculate reach 

# In[ ]:





# # shape for GAM 

# In[20]:


tel_df_raw = pd.read_csv(f"../data/interim/{gam_info['file_timeinfo']}_{platformID}_input.csv")
tel_df_raw = tel_df_raw.rename(columns={'Language Service': 'ServiceName'})
tel_df = tel_df_raw.merge(service_codes[['ServiceName', 'ServiceID']], on='ServiceName', how='left', indicator=True)
print(f'no mismatch:\n {tel_df._merge.value_counts()}')

tel_df = tel_df.drop(columns='_merge')

country_lookup = {"PER": "IRN",
                  "POR": "BRA",
                  "RUS": "RUS",
                  "UZB": "UZB",
                  "ARA": "UNK",
                  "UKR": "UKR" }
tel_df['PlaceID'] = tel_df['ServiceID'].map(country_lookup)

tel_df = tel_df.groupby(['w/c', 'PlaceID', 'ServiceID'])['Reach'].sum().reset_index()
tel_df.to_csv(f"../data/processed/{gam_info['file_timeinfo']}_{platformID}_uniqueViewer_country.csv",
              index=None)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
pd.set_option('display.float_format', '{:.00f}'.format)

import numpy as np
from sympy import symbols, solve, lambdify


# In[3]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules 
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[9]:


# service
service_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='ServiceID')

# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# social media accounts
dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts_all = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')


# # VK

# In[ ]:


platformID = 'VKO'



# # Weibo

# In[ ]:


platformID = 'WEI'


# # OK.RU

# In[ ]:


platformID = 'OKR'


# # Viber

# In[ ]:


platformID = 'VIB'


# # Whatsapp

# In[ ]:


platformID = 'WHA'


# In[ ]:





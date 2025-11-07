#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd


# ## import helper 

# In[2]:


platformID = 'FBE'


# In[3]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules
from functions import execute_sql_query
import test_functions

from config_GAM2025 import gam_info


# In[4]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID',
                             keep_default_na=False)

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period')
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# social media accounts

dtype_dict = {'Channel ID': 'str',
              'Linked FB Account': 'str'}
socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}", dtype=dtype_dict,
                                     sheet_name='Social Media Accounts new')

socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Year'] == gam_info['file_timeinfo']]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
#socialmedia_accounts['Channel ID'] = socialmedia_accounts['Channel ID'].dropna().apply(lambda x: str(int(x)))

channel_ids = socialmedia_accounts['Channel ID'].unique().tolist()
formatted_channel_ids = ', '.join(f"'{channel_id}'" for channel_id in channel_ids)


# ## country

# In[5]:


'''WHERE
        (fb_metric_id = 'page_impressions_by_country_unique' 
        AND 
        fb_page_id IN ({formatted_channel_ids})
        AND
        fb_metric_end_time Between  '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}')
        OR 
        (fb_metric_id = 'page_fans_country' 
        AND
        fb_page_id IN ({formatted_channel_ids})
        AND
        fb_metric_period = 'lifetime' 
        AND 
        fb_metric_end_time Between  '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}')
        '''
sql_query = f"""
    SELECT 
        fb_page_id, 
        fb_page_name, 
        fb_metric_id, 
        fb_metric_period, 
        fb_metric_breakdown, 
        fb_metric_end_time, 
        fb_metric_value 
    FROM
        redshiftdb.central_insights.fb_page_insights
    WHERE
        fb_metric_id = 'page_fans_country' 
        AND
        fb_metric_period = 'lifetime' 
        AND 
        fb_metric_end_time Between  '{gam_info['weekEnding_start']}' and '{gam_info['weekEnding_end']}'
        ;
    """
file = f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_country_redshift_extract.csv"

'''
df = execute_sql_query(sql_query)
df['fb_page_id'] = df['fb_page_id'].astype(str)
df.to_csv(file, index=False, na_rep='')
'''
facebook_country_raw = pd.read_csv(file, keep_default_na=False)
facebook_country_raw['fb_page_id'] = facebook_country_raw['fb_page_id'].astype(str)
#facebook_country_raw['fb_metric_end_time'] = pd.to_datetime(facebook_country_raw['fb_metric_end_time'])
# run tests
column_name = 'fb_metric_id'
test_functions.test_filter_elements_returned(facebook_country_raw,
                                             ['page_impressions_by_country_unique', 'page_fans_country'],
                                             column_name, "1_FB_7", "engagement sql query - metric test")
column_name = 'fb_page_id'
test_functions.test_filter_elements_returned(facebook_country_raw, channel_ids, column_name, 
                                             "1_FB_8", "engagement sql query - page test")

###
facebook_country_raw['fb_metric_end_time'] = pd.to_datetime(facebook_country_raw['fb_metric_end_time'])
week_tester['week_ending'] = pd.to_datetime(week_tester['week_ending'])

# Then rename and run the test
facebook_country_raw_renamed = facebook_country_raw.rename(columns={'fb_metric_end_time': 'week_ending'})
test_functions.test_weeks_presence('week_ending', facebook_country_raw_renamed, week_tester, '1_FB_9', "engagement sql query")
###
'''test_functions.test_weeks_presence('week_ending', 
                                    facebook_country_raw.rename(columns={'fb_metric_end_time': 'week_ending'}), 
                                    week_tester, '1_FB_9', "engagement sql query")
'''
print(facebook_country_raw.shape)
facebook_country_raw.sample(5)


# In[6]:


####  DANGER ZONE - doing weird stuff ####
# in minnie's dataset she substitute for a week that was missing in the dataset -  reproducing this for QA purposes only and wrote the code in a way that it won't affect any future queries
missing_date = "2024-04-21"
target_date = pd.to_datetime(missing_date)

if (target_date in week_tester.week_ending.values) and (target_date not in facebook_country_raw.fb_metric_end_time.values):
    substitute = facebook_country_raw[facebook_country_raw['fb_metric_end_time'] == "2025-01-05"]
    substitute['fb_metric_end_time'] = missing_date
    facebook_country_raw = pd.concat([facebook_country_raw, substitute])
facebook_country_raw[facebook_country_raw['fb_metric_end_time'] == missing_date]


# In[7]:


# Group by specified columns and sum the fb_metric_value
facebook_country_sum = facebook_country_raw.groupby(['fb_page_id', 'fb_metric_end_time', 'fb_metric_id'])\
                                        .agg(Sum_fb_metric_value=('fb_metric_value', 'sum'))\
                                        .reset_index()
facebook_country = facebook_country_raw.merge(facebook_country_sum, how='inner',
                                              on= ['fb_page_id', 'fb_metric_end_time', 'fb_metric_id'])
test_functions.test_inner_join(facebook_country_raw, facebook_country_sum, 
                               ['fb_page_id', 'fb_metric_end_time', 'fb_metric_id'], 
                               '1_FB_10', test_step='calculating country %')

facebook_country['country_%'] = facebook_country['fb_metric_value']/facebook_country['Sum_fb_metric_value']
facebook_country['fb_metric_end_time'] = pd.to_datetime(facebook_country['fb_metric_end_time']).dt.date
facebook_country.head()


# In[8]:


#page_impressions_by_country_unique only for half of the year
#upi_country = facebook_country[facebook_country['fb_metric_id'] == 'page_impressions_by_country_unique']
upi_country = facebook_country[facebook_country['fb_metric_id'] == 'page_fans_country']
upi_country = upi_country.rename(columns={'fb_metric_breakdown': 'YT-_FBE_codes', 
                                          'fb_metric_end_time': 'week_ending'})
upi_country['week_ending'] = pd.to_datetime(upi_country['week_ending']).dt.date
upi_country = upi_country.merge(country_codes[['YT-_FBE_codes', 'PlaceID']], on='YT-_FBE_codes', how='left',
                 indicator=True)


# In[13]:


'''
# country test
test_functions.test_country_percentage(upi_country, 
                        ['fb_page_id', 'fb_page_name', 'week_ending'], 
                        '1_FB_11', test_step='processing country sql data')

# join with GAM lookup table on w/c 
cols_left = ['fb_page_id', 'fb_page_name', 'fb_metric_id', 'fb_metric_period',
             'week_ending', 'fb_metric_value', 'Sum_fb_metric_value', 'country_%', 'Country Code']

# week test
test_functions.test_weeks_presence('week_ending', upi_country, week_tester, 
                                   '1_FB_12', test_step='processing country sql data')

# account test
column_name = 'fb_page_id'
test_functions.test_filter_elements_returned(upi_country_weeksTested, channel_ids, column_name, 
                              "1_FB_13", test_step='processing country sql data')
'''
# add week info 
cols_left = ['fb_page_id', 'fb_page_name', 'fb_metric_id', 'fb_metric_period',
             'week_ending', 'fb_metric_value', 'Sum_fb_metric_value', 'country_%', 'YT-_FBE_codes', 'PlaceID']

cols_right = ["week_ending" , "WeekNumber_finYear", "w/c"]
upi_country['week_ending'] = pd.to_datetime(upi_country['week_ending'])
upi_country_weeksTested = upi_country[cols_left].merge(week_tester[cols_right], how= 'inner',
                                                       on= 'week_ending' )

upi_country_weeksTested.sample()
upi_country_weeksTested.to_csv(f"../data/raw/{platformID}/{gam_info['file_timeinfo']}_{platformID}_REDSHIFT_COUNTRY.csv", 
                               index=None)


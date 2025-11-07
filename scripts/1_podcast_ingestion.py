#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd


# ## import helper

# In[2]:


import sys
from pathlib import Path

# Add ../helper to sys.path
helper_path = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(helper_path))

# Now import your modules
from config_GAM2025 import gam_info

from functions import execute_sql_query
import test_functions


# In[3]:


# country
country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='CountryID')

# week 
week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period', index_col=0)
week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])

# podcast details
podcast_details = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                                     sheet_name='Podcast').dropna(how='all')
podcast_details.sample()


# ## helper functions

# In[4]:


def get_formatted_values(df, query_type):
    values = df[df['Query Type'] == query_type]['Value'].tolist()
    return ', '.join(f"'{value}'" for value in values)

def generate_when_clauses(df):
    when_clause_brands = ''
    when_clause_programmes = ''
    for index, row in df.iterrows():
        if row['Query Type'] == 'brand_id':
            when_clause_brands += f"           WHEN vmb.master_brand_id = '{row['Value']}' THEN '{row['Service']}'\n"
        elif row['Query Type'] == 'program_title':
            when_clause_programmes += f"           WHEN vmb.programme_title = '{row['Value']}' THEN '{row['Service']}'\n"
    return when_clause_brands, when_clause_programmes


# In[5]:


def execute_sql_query_with_output(sql_query, file_timeinfo, output_filename):
    df = execute_sql_query(sql_query)
    if df is not None:
        display(df.head())
        
    df.to_csv(f"../data/raw/podcast/{file_timeinfo}_{output_filename}.csv", index=False)
    
    return df


# # ingestion

# ## test I find all the programes

# In[6]:


# Cell 1: Construct and execute the first query
formatted_brand_ids = get_formatted_values(podcast_details, 'brand_id')
formatted_program_titles = get_formatted_values(podcast_details, 'program_title')

sql_query_1 = f"""SELECT 
                    vmb.master_brand_id,
                    vmb.programme_title,
                    COUNT(prd.version_id) AS entry_count
                FROM 
                    redshiftdb.podcasts_rss_downloads.podcasting_raw_data prd
                INNER JOIN 
                    redshiftdb.prez.scv_vmb vmb 
                    ON prd.version_id = vmb.version_id
                WHERE 
                    prd.date_utc BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['weekEnding_end']}'
                    AND (
                        vmb.master_brand_id IN ({formatted_brand_ids})
                    OR 
                        vmb.programme_title IN ({formatted_program_titles})
                    )
                GROUP BY 
                    vmb.master_brand_id, vmb.programme_title
                HAVING 
                    COUNT(prd.version_id) > 0
                    ;"""

test_df = execute_sql_query_with_output(sql_query_1, 
                                        gam_info['file_timeinfo'], 
                                        'podcast_test_finding_all_brandAndProgrammes')
################################### Testing ################################### 
test_step = 'sql returns for all programmes'
column_name = 'master_brand_id'
brand_ids = [title.strip("'") for title in formatted_brand_ids.split(", ")]
test_functions.test_filter_elements_returned(test_df, brand_ids, column_name, "1_POD_1", test_step)
column_name = 'programme_title'
program_titles = [title.strip("'") for title in formatted_program_titles.split(", ")]
test_functions.test_filter_elements_returned(test_df, program_titles, column_name, '1_POD_2', test_step)

################################### Testing ################################### 


# ## Individual language services 

# In[7]:


#Construct and execute the second query
when_clause_brands, when_clause_programmes = generate_when_clauses(podcast_details)

sql_query_2 = f"""
    SELECT DATE_TRUNC('week', prd.date_utc) AS week,
           CASE
               {when_clause_programmes}
               ELSE CASE
                   {when_clause_brands}
                   ELSE 'Unknown'
               END
           END AS service,
           country,
               COUNT(DISTINCT CONCAT(prd.ip, prd.useragent)) AS uniques,
           COUNT(DISTINCT prd.ip) AS old_uniques,
           COUNT(*) AS downloads
    FROM redshiftdb.podcasts_rss_downloads.podcasting_raw_data prd
    INNER JOIN redshiftdb.prez.scv_vmb vmb 
        ON prd.version_id = vmb.version_id
    WHERE 
        prd.date_utc BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['weekEnding_end']}'
        AND (
        vmb.master_brand_id IN ({formatted_brand_ids})
        OR 
        vmb.programme_title IN ({formatted_program_titles})
        )
    GROUP BY week, service, country
;
"""

podcast_raw = execute_sql_query_with_output(sql_query_2, gam_info['file_timeinfo'], 'podcast_individualLanguages_redshift_extract')
podcast_raw.rename(columns={'week': 'w/c'}, inplace=True)
display(podcast_raw.sample())

################################### Testing ################################### 
test_step = 'sql returns for individual language services'
test_functions.podcast_test_services_in_results(podcast_raw, podcast_details, '1_POD_3', test_step)
test_functions.podcast_check_unknown_services(podcast_raw,'1_POD_4', test_step)

# weeks there? 
test_functions.test_weeks_presence_per_account('w/c', 'service', podcast_raw, week_tester,
                                                '1_POD_5', test_step)
################################### Testing ################################### 


# ## BBC World Service Languages
# 

# In[8]:


# Cell 3: Construct and execute the third query with WSL filter
total_wsl = podcast_details[podcast_details['* BBC World Service Languages'] == True]
formatted_brand_ids = get_formatted_values(total_wsl, 'brand_id')
formatted_program_titles = get_formatted_values(total_wsl, 'program_title')

sql_query_3 = f"""
    SELECT DATE_TRUNC('week', prd.date_utc) AS week,
           CASE
               WHEN 
                    vmb.master_brand_id IN ({formatted_brand_ids})
                    OR 
                    vmb.programme_title IN ({formatted_program_titles})
                    THEN '* BBC World Service Languages'
           END AS service,
           country,
           COUNT(DISTINCT CONCAT(prd.ip, prd.useragent)) AS uniques,
           COUNT(DISTINCT prd.ip) AS old_uniques,
           COUNT(*) AS downloads
    FROM redshiftdb.podcasts_rss_downloads.podcasting_raw_data prd
    INNER JOIN redshiftdb.prez.scv_vmb vmb 
        ON prd.version_id = vmb.version_id
    WHERE 
        prd.date_utc BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['weekEnding_end']}'
        AND (
        vmb.master_brand_id IN ({formatted_brand_ids})
        OR 
        vmb.programme_title IN ({formatted_program_titles})
        )
    GROUP BY week, service, country
;
"""

podcast_total_wsl = execute_sql_query_with_output(sql_query_3, gam_info['file_timeinfo'], 'podcast_totalWSL_redshift_extract')
podcast_total_wsl.rename(columns={'week': 'w/c'}, inplace=True)
display(podcast_total_wsl.sample())
#wpodcast_test_services_in_results(podcast_total_wsl, total_wsl)
################################### Testing ################################### 
test_step = 'sql returns for WSL'
test_functions.podcast_check_unknown_services(podcast_total_wsl,'1_POD_6', test_step)

test_functions.test_weeks_presence_per_account('w/c', 'service', podcast_total_wsl, week_tester,
                                                '1_POD_7', test_step)
################################### Testing ################################### 


# ## BBC World Service 

# In[9]:


# Filter the DataFrame based on '* BBC World Service'
total_ws = podcast_details[podcast_details['* BBC World Service'] == True]

# Generate the formatted brand IDs and program titles
formatted_brand_ids = get_formatted_values(total_ws, 'brand_id')
formatted_program_titles = get_formatted_values(total_ws, 'program_title')

# Construct the SQL query
sql_query = f"""
    SELECT DATE_TRUNC('week', prd.date_utc) AS week,
           CASE
               WHEN 
                    vmb.master_brand_id IN ({formatted_brand_ids})
                    OR 
                    vmb.programme_title IN ({formatted_program_titles})
                    THEN '* BBC World Service'
                WHEN country!='gb' 
                    THEN 'UKPS'
           ELSE 'UKPS GB'      
           END AS service,
           country,
           COUNT(DISTINCT CONCAT(prd.ip, prd.useragent)) AS uniques,
           COUNT(DISTINCT prd.ip) AS old_uniques,
           COUNT(*) AS downloads
    FROM redshiftdb.podcasts_rss_downloads.podcasting_raw_data prd
    INNER JOIN redshiftdb.prez.scv_vmb vmb 
        ON prd.version_id = vmb.version_id
    WHERE 
        prd.date_utc BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['weekEnding_end']}'
    GROUP BY week, service, country
    HAVING service != 'UKPS GB'
;
"""

podcast_total_ws = execute_sql_query_with_output(sql_query, gam_info['file_timeinfo'], 'podcast_totalWS_redshift_extract')
podcast_total_ws.rename(columns={'week': 'w/c'}, inplace=True)
display(podcast_total_wsl.sample())

################################### Testing ################################### 
test_step = 'sql returns for WS'
#podcast_test_services_in_results(podcast_total_ws, total_ws)
test_functions.podcast_check_unknown_services(podcast_total_ws, '1_POD_8', test_step)

test_functions.test_weeks_presence_per_account('w/c', 'service', podcast_total_wsl, week_tester,
                                                '1_POD_9', test_step)
################################### Testing ################################### 


# ## all BBC 

# In[10]:


# Filter the DataFrame based on '* BBC World Service'
all_bbc = podcast_details[podcast_details['* All BBC'] == True]

# Generate the formatted brand IDs and program titles
formatted_brand_ids = get_formatted_values(all_bbc, 'brand_id')
formatted_program_titles = get_formatted_values(all_bbc, 'program_title')

# Construct the SQL query
sql_query = f"""
    SELECT DATE_TRUNC('week', prd.date_utc) AS week,
           CASE
               WHEN 
                    vmb.master_brand_id IN ({formatted_brand_ids})
                    OR 
                    vmb.programme_title IN ({formatted_program_titles})
                    THEN '* All BBC'
               WHEN country!='gb' THEN '* All BBC'  
           END AS service,
           country,
           COUNT(DISTINCT CONCAT(prd.ip, prd.useragent)) AS uniques,
           COUNT(DISTINCT prd.ip) AS old_uniques,
           COUNT(*) AS downloads
    FROM redshiftdb.podcasts_rss_downloads.podcasting_raw_data prd
    INNER JOIN redshiftdb.prez.scv_vmb vmb 
        ON prd.version_id = vmb.version_id
    WHERE 
        prd.date_utc BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['weekEnding_end']}'
    GROUP BY week, service, country
;
"""

podcast_total_allBBC = execute_sql_query_with_output(sql_query, gam_info['file_timeinfo'], 'podcast_allBBC_redshift_extract')
podcast_total_allBBC.rename(columns={'week': 'w/c'}, inplace=True)
display(podcast_total_wsl.sample())

################################### Testing ################################### 
test_step = 'sql returns for all BBC'

#podcast_test_services_in_results(podcast_total_allBBC, all_bbc)
test_functions.podcast_check_unknown_services(podcast_total_allBBC, '1_POD_10', test_step)

test_functions.test_weeks_presence_per_account('w/c', 'service', podcast_total_allBBC, week_tester,
                                                '1_POD_11', test_step)

################################### Testing ################################### 


# In[ ]:





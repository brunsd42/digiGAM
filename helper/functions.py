from IPython.display import display

import os
import pandas as pd 
import numpy as np
import psycopg2
import urllib
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import missingno as msno

import urllib.parse

import security_config
import test_functions
    
################### PIANO
def convert_url_to_query(url, start, end):
    # Extract the query parameter from the URL
    parsed_url = urllib.parse.urlparse(url)
    query_param = urllib.parse.parse_qs(parsed_url.query).get('param', [None])[0]
    
    if query_param:
        # Decode the JSON string
        decoded_param = urllib.parse.unquote(query_param)
        query_dict = json.loads(decoded_param)

        query_dict['period']['p1'][0]['start'] = start
        query_dict['period']['p1'][0]['end'] = end

        # Remove options if empty
        if not query_dict.get('options'):
            query_dict['options'] = {}
        
        return query_dict
    else:
        return None

def fetch_data(page_num, data, api_query_key):
    api_endpoint = 'https://api.atinternet.io/v3/data/getData'

    # Define the request headers with the API key
    headers = {
        'x-api-key': api_query_key
    }
    
    # Create a session and configure retry strategy
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    """Fetch data from a specific page number."""
    data["page-num"] = page_num
    response = session.post(api_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def api_call(data, api_query_key):
    
    # Initialize a list to store all rows of data
    all_data_records = []
    page_num = 1
    
    while True:   
        response_json = fetch_data(page_num, data, api_query_key)
        
        if response_json:
            data_records = response_json.get('DataFeed', {}).get('Rows', [])
            if not data_records:
                break  # Exit loop if no more data is returned
            
            all_data_records.extend(data_records)
            page_num += 1  # Move to the next page
        else:
            break  # Exit loop if there was an error
        
    
    # Convert all data into a pandas DataFrame
    return pd.DataFrame(all_data_records)

######################## REDSHIFT 
def execute_sql_query(sql_query):
    host = security_config.REDSHIFT_HOST
    port = security_config.REDSHIFT_PORT
    user = security_config.REDSHIFT_USER
    password = security_config.REDSHIFT_PASS
    dbname = security_config.REDSHIFT_DB

    conn_string = f"host={host} port={port} user={user} password={password} dbname={dbname}"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])     
        return df
    
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None

    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

######################## lookup file
def lookup_loader(gam_info, platformID, with_country=False, country_col=''):
    # week 
    week_cols = ['w/c']
    week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", 
                                sheet_name='GAM Period')
    week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
    
    today = pd.Timestamp.today().normalize()
    last_monday = today - pd.Timedelta(days=(today.weekday() % 7))
    week_tester = week_tester[week_tester['w/c'] < last_monday]

    test_functions.test_lookup_files(week_tester, ['w/c'], 
                                     [f"{platformID}_1c_0", 
                                      f"{platformID}_1c_1", 
                                      f"{platformID}_1c_2"], 
                                     test_step = "lookup files - ensuring week tester is correct")
    # social media accoutns
    channel_cols=['Channel ID']
    dtype_dict = {'Channel ID': 'str',
                  'Linked FB Account': 'str'}
    socialmedia_accounts = pd.read_excel(f"../../{gam_info['lookup_file']}",
                                         dtype=dtype_dict,
                                         sheet_name='Social Media Accounts new')
    
    socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['PlatformID'] == platformID]
    socialmedia_accounts = socialmedia_accounts[socialmedia_accounts['Status'] == 'active']
    socialmedia_accounts['Channel ID'] = platformID + socialmedia_accounts['Channel ID']
    socialmedia_accounts['Start'] = pd.to_datetime(socialmedia_accounts['Start'], 
                                                   errors='coerce', dayfirst=True)
    socialmedia_accounts['End'] = pd.to_datetime(socialmedia_accounts['End'], 
                                                   errors='coerce', dayfirst=True)
    test_functions.test_lookup_files(socialmedia_accounts, ['Channel ID'], 
                                     [f"{platformID}_1c_3", 
                                      f"{platformID}_1c_4", 
                                      f"{platformID}_1c_5"],
                                     test_step = "lookup files - ensuring social media accounts is correct")
    
    # country
    if with_country:
        country_cols = [country_col, 'PlaceID']
        country_codes = pd.read_excel(f"../../{gam_info['lookup_file']}",
                                      sheet_name='CountryID',
                                      keep_default_na=False)[country_cols]
        
        test_functions.test_lookup_files(country_codes, country_cols, 
                                         [f"{platformID}_1c_6", 
                                          f"{platformID}_1c_7", 
                                          f"{platformID}_1c_8"],
                                         test_step="lookup files - ensuring country codes is correct")
        return {'week_tester': week_tester,
                'socialmedia_accounts': socialmedia_accounts,
                'country_codes': country_codes,
               }
    else:
        return {'week_tester': week_tester,
                'socialmedia_accounts': socialmedia_accounts,
               }


################################ single platform calculations

def calculate_rolling_avg_country_split(df, metric_col='rescaled_percentage', min_week=None, max_week=None):
    """
    For each channel and place, generate a full Monday calendar from min_week to max_week (inclusive),
    and compute the prev-4-week rolling average (shifted by 1) for every week, even if missing in original data.
    The metric column contains the country split (% per country).
    """
    # Convert dates
    min_week = pd.to_datetime(min_week)
    max_week = pd.to_datetime(max_week)

    # Ensure w/c is datetime and sort
    df = df.copy()
    df['w/c'] = pd.to_datetime(df['w/c'])
    df = df.sort_values(['Channel ID', 'PlaceID', 'w/c'])
    df = df.groupby(['Channel ID', 'PlaceID', 'w/c'])[metric_col].sum().reset_index()

    # Full Monday calendar
    calendar = pd.date_range(start=min_week, end=max_week, freq='7D')

    results = []

    for (ch, place), grp in df.groupby(['Channel ID', 'PlaceID']):
        # Reindex to full calendar
        grp = grp.set_index('w/c').reindex(calendar)
        grp.index.name = 'w/c'

        # Compute rolling average of last 4 observed weeks (excluding current)
        grp[metric_col] = (
            grp[metric_col]
            .shift(1)
            .rolling(window=4, min_periods=1)
            .mean()
        )

        # Add identifiers
        grp['Channel ID'] = ch
        grp['PlaceID'] = place

        results.append(grp[['Channel ID', 'PlaceID', metric_col]].reset_index())

    # Combine all
    result = pd.concat(results, ignore_index=True)
    return result[['Channel ID', 'PlaceID', 'w/c', metric_col]]


def gnl_expander(df):
    """
    Duplicate rows where ServiceID == 'GNL' into two rows with ServiceID 'BNI' and 'BNO'.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'ServiceID' column.

    Returns:
        pd.DataFrame: DataFrame with additional rows for BNI and BNO.
    """
    gnl_rows = df[df['ServiceID'] == 'GNL']
    if not gnl_rows.empty:
        bni_rows = gnl_rows.copy()
        bni_rows['ServiceID'] = 'BNI'

        bno_rows = gnl_rows.copy()
        bno_rows['ServiceID'] = 'BNO'

        df = pd.concat([df, bni_rows, bno_rows], ignore_index=True)

    return df


def filter_channels_by_weeks(df, week_col='w/c', channel_col='Channel ID', min_weeks=12):
    """
    Filters channels that have at least `min_weeks` of data and reports excluded channels.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing weekly data.
    week_col : str
        Column name representing the week (default: 'w/c').
    channel_col : str
        Column name representing the channel ID (default: 'Channel ID').
    min_weeks : int
        Minimum number of weeks required to keep a channel (default: 12).

    Returns:
    -------
    filtered_df : pd.DataFrame
        DataFrame containing only channels with >= min_weeks.
    excluded_summary : pd.DataFrame
        DataFrame listing excluded channels and their week counts.
    """
    # Count unique weeks per channel
    weeks_per_channel = df.groupby(channel_col)[week_col].nunique()

    # Separate valid and excluded channels
    valid_channels = weeks_per_channel[weeks_per_channel >= min_weeks].index
    excluded_channels = weeks_per_channel[weeks_per_channel < min_weeks]

    # Create summary DataFrame for excluded channels
    excluded_summary = excluded_channels.reset_index().rename(columns={week_col: 'weeks_count'})
    print('Channels that are excluded due to number of weeks they had audiences:')
    display(excluded_summary)
    # Filter original DataFrame
    filtered_df = df[df[channel_col].isin(valid_channels)]

    return filtered_df
    
def include_uk_decision(df, lookup):
    temp = df.merge(lookup[['Channel ID', 'Excluding UK']], on=['Channel ID'] , how='left')
    return temp[~((temp['PlaceID']=='UK') & (temp['Excluding UK']=='Yes'))]
        

'''def calculate_annualy(df, platform, gam_info, aggregation_pattern='year'):
    world_avg = df.groupby(['ServiceID', 'w/c'])['Reach'].sum().reset_index()
    world_avg = world_avg.groupby('ServiceID')['Reach'].mean().reset_index()
    world_avg['PlaceID'] = 'Total'
    #display(world_avg)

    if aggregation_pattern == 'year':
        year_avg = df.groupby(['ServiceID', 'PlaceID'])['Reach'].sum().reset_index()
        year_avg['Reach'] =  year_avg['Reach']/gam_info['number_of_weeks']
        #display(year_avg.groupby('Service Code')['Reach'].sum().reset_index())
    else: 
        print('calculating the average and not divding by 52')
        year_avg = df.groupby(['ServiceID', 'PlaceID', 'w/c'])['Reach'].sum().reset_index()
        year_avg = year_avg.groupby(['ServiceID', 'PlaceID'])['Reach'].mean().reset_index()
        
    annual_df = pd.concat([world_avg, year_avg], )
    annual_df['PlatformID'] = platform
    annual_df['YearGAE'] = gam_info['YearGAE']
    return annual_df'''

def calculate_annualy(df, platformID, gam_info, aggregation_type='new'):
    # Calculate world average
    world_avg = df.groupby(['ServiceID', 'w/c'])['Reach'].sum().reset_index()
    world_avg = world_avg.groupby('ServiceID')['Reach'].mean().reset_index()
    world_avg['PlaceID'] = 'Total'

    if aggregation_type == 'new':
        print('calculating annual by the new method')
        df['w/c'] = pd.to_datetime(df['w/c'])
        week_tester = pd.read_excel(f"../../{gam_info['lookup_file']}", sheet_name='GAM Period',)
        week_tester['w/c'] = pd.to_datetime(week_tester['w/c'])
        df = df.merge(week_tester[['w/c', 'WeekNumber_finYear']], on='w/c', how='left')
        
        def compute_reach(group):
            weeks = group['WeekNumber_finYear'].drop_duplicates().sort_values()
            if len(weeks) <= 12:
                
                start = weeks.min()
                expected_weeks = pd.DataFrame({'WeekNumber_finYear': list(range(start, start + len(weeks)))})
                merged = expected_weeks.merge(group[['WeekNumber_finYear']], 
                                              on='WeekNumber_finYear', 
                                              how='inner')
        
                if len(merged) == len(weeks):
                    return group['Reach'].mean()
                else:
                    # given that the dataset will always contain the financial year data we can take the 
                    # number of weeks from that column (it starts with 1 and increaeses week by week)
                    return group['Reach'].sum() / weeks.max()
    
            return group['Reach'].sum() / weeks.max()
    
        # Apply the logic to each ServiceID-PlaceID group
        year_avg = df.groupby(['ServiceID', 'PlaceID']).apply(compute_reach).reset_index(name='Reach')
    else: 
        print('calculating the average and not divding by weeks in the timeframe')
        year_avg = df.groupby(['ServiceID', 'PlaceID', 'w/c'])['Reach'].sum().reset_index()
        year_avg = year_avg.groupby(['ServiceID', 'PlaceID'])['Reach'].mean().reset_index()
    
    # Combine world and year averages
    annual_df = pd.concat([world_avg, year_avg], ignore_index=True)
    annual_df['PlatformID'] = platformID
    annual_df['YearGAE'] = gam_info['YearGAE']

    return annual_df

def summary_excel(df, business_part, platformID, gam_info, aggregation_type='new',
                  store_annual=False):
    weekly_df = df.copy()
    weekly_df = weekly_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    weekly_df['PlatformID'] = platformID
    weekly_df['YearGAE'] = gam_info['YearGAE']
    
    path = f"../data/singlePlatform/{platformID}/weekly/"
    os.makedirs(path, exist_ok=True)
    file = f"{gam_info['file_timeinfo']}_WEEKLY_{platformID}_{business_part}byCountry.xlsx"
    
    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'w/c', 'Reach']
    weekly_df[col_order].to_excel(f"{path}/{file}", index=None)
    print(f"saved weekly file for {business_part} as:\n {path}/{file}") 

    if store_annual:
        col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'Reach']
        annual_df = calculate_annualy(weekly_df, platform, gam_info, aggregation_type)[col_order]
        
        file_path = f"{gam_info['file_timeinfo']}_{platform}_{business_part}.xlsx"
        annual_df.to_excel(path+file_path, index=None)
        print(f"saved annual file for {business_part} as:\n {file_path}")

    return weekly_df
    
'''def calculating_weekly_reach(df, platform, bu, gam_info, service=''):
    # calculate weekly reach by service, country and week for a given platform 
    df = df.groupby(['ServiceID', 'Country', 'Week Number'])['Engaged Users by Country'].sum().reset_index()
    df['YearGAE'] = gam_info['YearGAE']
    df['PlatformID'] = platform

    df.to_excel(f"../data/singlePlatform/output/WEEKLY_{platform}_{bu}_{service}.xlsx")
    return df
'''    
'''def calculating_annual_reach(df, platform, bu, gam_info, service=''):
    # calculate weekly global reach by service for a given platform 
    df_global = df.groupby(['ServiceID', 'PlatformID', 'Week Number'])['Engaged Users by Country'].sum().reset_index()
    df_global['Country'] = '* Total'
    
    # combine weekly by country and global reach
    weekly_reach = pd.concat([df, df_global])
    
    # calculate annual reach by country and service
    avg_annual_reach = weekly_reach.groupby(['ServiceID', 'Country', 'PlatformID'])['Engaged Users by Country'].sum().reset_index(name='Reach')
    avg_annual_reach['Reach'] = avg_annual_reach['Reach']/gam_info['number_of_weeks']
    
    avg_annual_reach.to_excel(f"../data/output/{platform}_{bu}_{service}.xlsx")

    return avg_annual_reach'''
'''
def sainsbury_formula(df, population_col, channel_list, col_name):
    """
    Apply the sainsforumula 

    Parameters:

    Returns:
    bool: True if the test passes, False otherwise.

    Example
    
    """
    def calculate_formula(row):
        population = row[population_col]
        product = 1
        for channel in channel_list:
            product *= (1 - (row[channel] / population))
        return (1 - product) * population
    
    df[col_name] = df.apply(calculate_formula, axis=1)
    return df
'''
def sainsbury_formula(df, population_col, channel_list, col_name, deal_with_zero=False):
    """
    Apply the Sainsbury formula with optional shortcut logic.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    population_col (str): Column name for population.
    channel_list (list): List of two channel column names.
    col_name (str): Name of the output column.
    use_shortcut (bool): If True, use the shortcut when one channel is 0.

    Returns:
    pd.DataFrame: DataFrame with the new column added.
    """
    
        
    def calculate_formula(row):
        population = row[population_col]
        channel_values = [row[channel] for channel in channel_list]

        if deal_with_zero and len(channel_list) == 2:
            if channel_values[0] == 0:
                return channel_values[1]
            elif channel_values[1] == 0:
                return channel_values[0]

        product = 1
        for value in channel_values:
            product *= (1 - (value / population))
        return (1 - product) * population

    df[col_name] = df.apply(calculate_formula, axis=1)
    return df

def calculate_weekly_sumServices(df, serviceID, platformID, gam_info):
    df = df.copy()
    # temporary to explain discrepancy to minnie's values
    df_weekly = df.groupby(['PlaceID', 'w/c'])['Reach'].sum().reset_index()
    df_weekly['Reach'] = df_weekly['Reach']
    df_weekly['ServiceID'] = serviceID
    df_weekly['PlatformID'] = platformID
    df_weekly['YearGAE'] = gam_info['YearGAE']
    
    path = f"../data/singlePlatform/{platformID}/weekly/"
    filename = f"{gam_info['file_timeinfo']}_WEEKLY_{platformID}_{serviceID}byCountry.xlsx"
    
    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'w/c', 'Reach']
    df_weekly.to_excel(path+filename, index=None)
    
    return df_weekly

def calculate_weekly_Services(df, serviceID, platformID, pop_size_col, gam_info, combi_type='sainsbury', ):
    
    if len(df) == 0:
        print('no data in the dataframe')
        return df
    df = df.copy()
    if combi_type == 'sainsbury':
        service_list = df.ServiceID.unique()
        if len(service_list) >= 2:
            df_weekly = pd.crosstab(index = [ df['PlaceID'], 
                                              df[pop_size_col], 
                                              df['w/c']],
                                    columns = df['ServiceID'],
                                    values =  df['Reach'],
                                    aggfunc='sum'
                                    ).reset_index().fillna(0)
            df_weekly = sainsbury_formula(df_weekly, pop_size_col, service_list, 'Reach')
        else:
            df_weekly = df
    else:
        df_weekly = df.groupby(['PlaceID', 'w/c'])['Reach'].sum().reset_index()
    df_weekly['Reach'] = df_weekly['Reach']
    df_weekly['ServiceID'] = serviceID
    df_weekly['PlatformID'] = platformID
    df_weekly['YearGAE'] = gam_info['YearGAE']
    
    path = f"../data/singlePlatform/{platformID}/weekly/"
    filename = f"{gam_info['file_timeinfo']}_WEEKLY_{platformID}_{serviceID}byCountry.xlsx"
    
    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'w/c', 'Reach']
    df_weekly.to_excel(path+filename, index=None)
    
    return df_weekly

def process_overlap(data, service1, service2, grouped_service,
                    overlap_type, overlap_service_id, platformID, gam_info, path,
                    country_codes, pop_size_col, overlaps='n/a'):
    """
    Combines two services into a grouped service, applying overlap logic if both are present.
    If only one service has data, it is used directly.
    """

    # Ensure the grouped_service key exists
    if grouped_service not in data:
        data[grouped_service] = {}

    # Extract weekly data safely
    df1 = data.get(service1, {}).get('weekly', pd.DataFrame())
    df2 = data.get(service2, {}).get('weekly', pd.DataFrame())

    # Determine which services are available
    has_df1 = isinstance(df1, pd.DataFrame) and not df1.empty
    has_df2 = isinstance(df2, pd.DataFrame) and not df2.empty

    if not has_df1:
        print(f"Warning: {service1} weekly data is missing or empty.")
    if not has_df2:
        print(f"Warning: {service2} weekly data is missing or empty.")

    # If both are empty, skip
    if not has_df1 and not has_df2:
        print(f"No data to process for {service1} and {service2}. Skipping.")
        return pd.DataFrame(), pd.DataFrame()

    # If only one is available, use it directly
    if has_df1 and not has_df2:
        print(f"Only {service1} available. Using it as {grouped_service}.")
        pivot_df = df1.copy()
        pivot_df['ServiceID'] = grouped_service

    elif has_df2 and not has_df1:
        print(f"Only {service2} available. Using it as {grouped_service}.")
        pivot_df = df2.copy()
        pivot_df['ServiceID'] = grouped_service

    else:
        # Both available — proceed with overlap logic
        combined_df = pd.concat([df1, df2], ignore_index=True)

        pivot_df = pd.crosstab(
            index=[combined_df['PlaceID'], combined_df['w/c']],
            columns=combined_df['ServiceID'],
            values=combined_df['Reach'],
            aggfunc='sum'
        ).reset_index()

        # Ensure both service columns exist
        for service in [service1, service2]:
            if service not in pivot_df.columns:
                print(f"Warning: {service} column missing from pivot table. Filling with 0.")
                pivot_df[service] = 0
            else:
                pivot_df[service] = pivot_df[service].fillna(0)

        if overlap_type != 'sainsbury':
            overlap_df = overlaps[overlaps['Overlap Type'] == overlap_type]
            overlap_value = overlap_df.loc[overlap_df['ServiceID'] == overlap_service_id, 'overlap_%'].values[0]
            print(f"overlap applied: {overlap_value}")
            pivot_df['overlap'] = overlap_value

            pivot_df['Reach'] = pivot_df.apply(
                lambda row: row[service1] + row[service2] * (1 - row['overlap']) 
                if row[service1] > row[service2] 
                else row[service1] * (1 - row['overlap']) + row[service2],
                axis=1
            )
        else:
            pivot_df = pivot_df.merge(country_codes, on='PlaceID', how='left', indicator=True)
            print(f"adding population: {pivot_df._merge.value_counts()}")
            pivot_df = pivot_df.drop(columns=['_merge'])

            services = [service1, service2]
            pivot_df = sainsbury_formula(pivot_df, pop_size_col, services, 'Reach')

        pivot_df['ServiceID'] = grouped_service

    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)

    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = calculate_weekly_sumServices(pivot_df, grouped_service, platformID, gam_info)
    annual_df = calculate_annualy(data[grouped_service]['weekly'], platformID, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df

    return pivot_df, annual_df
    
def process_overlap_v2(data, service1, service2, grouped_service,
                       overlaps, overlap_type, overlap_service_id, platformID, gam_info, path,
                       country_codes, pop_size_col, service3=None):  
    """
    Combines services into a grouped service, applying overlap logic if applicable.
    If only one or two services are available, uses them directly.
    service3 is only used for overlap_type 'sainsbury'.
    """

    # Ensure the grouped_service key exists
    if grouped_service not in data:
        data[grouped_service] = {}

    # Extract weekly data safely
    def safe_get(service):
        entry = data.get(service, {})
        return entry['weekly'] if isinstance(entry, dict) and 'weekly' in entry else pd.DataFrame()

    df1 = safe_get(service1)
    df2 = safe_get(service2)
    df3 = safe_get(service3) if service3 else pd.DataFrame()

    # Track which services are available
    services = []
    combined_parts = []

    for svc, df in zip([service1, service2, service3], [df1, df2, df3]):
        if svc and isinstance(df, pd.DataFrame) and not df.empty:
            services.append(svc)
            combined_parts.append(df)
        elif svc:
            print(f"Warning: {svc} weekly data is missing or empty.")

    if not combined_parts:
        print(f"No data to process for {service1}, {service2}, {service3}. Skipping.")
        return pd.DataFrame(), pd.DataFrame()

    # Combine available data
    combined_df = pd.concat(combined_parts, ignore_index=True)

    # Pivot
    pivot_df = pd.crosstab(
        index=[combined_df['PlaceID'], combined_df['w/c']],
        columns=combined_df['ServiceID'],
        values=combined_df['Reach'],
        aggfunc='sum'
    ).reset_index()

    # Fill missing service columns with 0
    for service in services:
        if service not in pivot_df.columns:
            print(f"Warning: {service} column missing from pivot table. Filling with 0.")
            pivot_df[service] = 0
        else:
            pivot_df[service] = pivot_df[service].fillna(0)

    # Overlap logic
    if overlap_type != 'sainsbury':
        if grouped_service == 'EN2':
            pivot_df['Reach'] = np.where(
                (pivot_df.get('GNL', 0) + pivot_df.get('WSE', 0)) > pivot_df.get('WOR', 0),
                (pivot_df.get('GNL', 0) + pivot_df.get('WSE', 0)) + (0.892857142857143 * pivot_df.get('WOR', 0)),
                pivot_df.get('WOR', 0) + ((pivot_df.get('GNL', 0) + pivot_df.get('WSE', 0)) * 0.952380952380952)
            )
        elif len(services) >= 2:
            overlap_df = overlaps[overlaps['Overlap Type'] == overlap_type]
            overlap_value = overlap_df.loc[overlap_df['ServiceID'] == overlap_service_id, 'overlap_%'].values[0]
            print(f"overlap applied: {overlap_value}")
            pivot_df['overlap'] = overlap_value

            s1, s2 = services[:2]
            pivot_df['Reach'] = pivot_df.apply(
                lambda row: row[s1] + row[s2] * (1 - row['overlap']) 
                if row[s1] > row[s2] 
                else row[s1] * (1 - row['overlap']) + row[s2],
                axis=1
            )
        else:
            # Only one service available, use it directly
            pivot_df['Reach'] = pivot_df[services[0]]

    else:
        # Sainsbury logic
        pivot_df = pivot_df.merge(country_codes, on='PlaceID', how='left', indicator=True)
        print(f"adding population: {pivot_df._merge.value_counts()}")
        pivot_df = pivot_df.drop(columns=['_merge'])

        pivot_df = sainsbury_formula(pivot_df, pop_size_col, services, 'Reach')

    # Assign grouped service
    pivot_df['ServiceID'] = grouped_service

    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)

    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = calculate_weekly_sumServices(pivot_df, grouped_service, platformID, gam_info)
    annual_df = calculate_annualy(data[grouped_service]['weekly'], platformID, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platformID}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df

    return pivot_df, annual_df

def calculate_aggregated_services(data, stages, platform, gam_info, path, 
                                  overlaps, country_codes, pop_size_col):

    for stage in stages:
        grouped_service, s1, s2, o_type, o_id, use_v2, s3 = stage
        data[grouped_service] = {'weekly': pd.DataFrame(), 'annual': pd.DataFrame()}
        
        if use_v2:
            pivot, annual = process_overlap_v2(
                data=data,
                service1=s1,
                service2=s2,
                grouped_service=grouped_service,
                overlaps=overlaps,
                overlap_type=o_type,
                overlap_service_id=o_id,
                platformID=platform,
                gam_info=gam_info,
                path=path,
                country_codes=country_codes,
                pop_size_col=pop_size_col,
                service3=s3
            )
        else:
            pivot, annual = process_overlap(
                data=data,
                service1=s1,
                service2=s2,
                grouped_service=grouped_service,
                overlaps=overlaps,
                overlap_type=o_type,
                overlap_service_id=o_id,
                platformID=platform,
                gam_info=gam_info,
                path=path,
                country_codes=country_codes,
                pop_size_col=pop_size_col
            )
        
        data[grouped_service] = {
            'weekly': pivot,
            'annual': annual
        }

    return data


########################### COMPARING 


def compare_or_update_reference(df, reference_path, cols, update=False):
    """
    Compare DataFrame to a reference file (Pickle for accuracy).
    If update=True, overwrite the reference file with the new DataFrame.
    
    Parameters:
    - df: DataFrame to validate
    - reference_path: Path to reference Pickle file
    - cols: List of columns to compare
    - update: If True, update the reference file with df
    """
    # Select relevant columns
    df = df[cols].copy()

    # Normalize: sort rows and columns
    df = df.sort_values(by=cols).reset_index(drop=True)

    if update or not pd.io.common.file_exists(reference_path):
        df.to_pickle(reference_path)
        print(f"✅ Reference file updated at {reference_path}")
        return

    # Load reference
    ref_df = pd.read_pickle(reference_path)
    ref_df = ref_df.sort_values(by=cols).reset_index(drop=True)

    # Compare
    if df.equals(ref_df):
        print("✅ Output matches reference.")
    else:
        print("❌ Output differs from reference!")
        # Show sample differences
        print(df.compare(ref_df).head(10))
        
# Utility functions
def load_excel(path, sheet_name=None):
    if sheet_name != None: 
        return pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
    else:
        return pd.read_excel(path, engine='openpyxl')

def load_csv(path):
    return pd.read_csv(path)

def standardize_country_codes(df, column='Country Code'):
    return df.replace({column: {'WLF': 'WFI', '* Total': 'Total'}})

def run_comparison(original_df, new_df, column_mapping, key_columns, method='integer', threshold=0.0001):
    if method == 'integer':
        return compare_dataframes_integer(original_df, new_df, column_mapping, key_columns)
    elif method == 'percentage':
        return compare_dataframes_percentage(original_df, new_df, column_mapping, key_columns, threshold)
    else:
        raise ValueError("Unknown comparison method")

def compare_dataframes_integer(original_df, new_df, column_mapping, key_columns_new):
    """
    Compare two DataFrames and return rows that are missing or different.

    Parameters:
    - original_df: DataFrame from the original source
    - new_df: DataFrame from the new source
    - column_mapping: dict mapping original_df column names to new_df column names
    - key_columns_new: list of key columns using new_df naming

    Returns:
    - missing_from_new: rows in original_df not found in new_df
    - differing_rows: rows where key matches but mapped columns differ
    """

    # Rename original_df to match new_df column names
    original_df_renamed = original_df.rename(columns=column_mapping)

    # Ensure all required columns exist
    all_columns = list(column_mapping.values())
    original_subset = original_df_renamed[all_columns].copy()
    new_subset = new_df[all_columns].copy()

    # Round numeric columns to nearest integer
    for col in all_columns:
        if pd.api.types.is_numeric_dtype(original_subset[col]) and pd.api.types.is_numeric_dtype(new_subset[col]):
            original_subset[col] = original_subset[col].round(0).astype('Int64')
            new_subset[col] = new_subset[col].round(0).astype('Int64')
    
    # Merge to find differences
    merged = pd.merge(
        original_subset,
        new_subset,
        on=key_columns_new,
        how='outer',
        suffixes=('_orig', '_new'),
        indicator=True
    )

    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']

    # Differing rows: same keys but different values
    comparison_cols = [col for col in all_columns if col not in key_columns_new]

    # Fill NA with a sentinel value for comparison
    sentinel = -999999
    orig_comp = merged[[f"{col}_orig" for col in comparison_cols]].fillna(sentinel)
    new_comp  = merged[[f"{col}_new" for col in comparison_cols]].fillna(sentinel)
    
    differing_rows = merged[
    (merged['_merge'] == 'both') &
    (orig_comp.values != new_comp.values).any(axis=1)
    ]
    '''differing_rows = merged[
        (merged['_merge'] == 'both') &
        merged[[f"{col}_orig" for col in comparison_cols]].ne(
            merged[[f"{col}_new" for col in comparison_cols]].values
        ).any(axis=1)
        ]'''

    return missing_from_new, differing_rows

def compare_dataframes_percentage(original_df, new_df, column_mapping, key_columns_new, threshold=0.0001):
    """
    Compare two DataFrames and return rows that are missing or have percentage differences.

    Parameters:
    - original_df: DataFrame from the original source
    - new_df: DataFrame from the new source
    - column_mapping: dict mapping original_df column names to new_df column names
    - key_columns_new: list of key columns using new_df naming
    - threshold: minimum absolute difference to consider as significant

    Returns:
    - missing_from_new: rows in original_df not found in new_df
    - differing_rows: rows where key matches but mapped columns differ beyond threshold
    """

    # Rename original_df to match new_df column names
    original_df_renamed = original_df.rename(columns=column_mapping)

    # Ensure all required columns exist
    all_columns = list(column_mapping.values())
    original_subset = original_df_renamed[all_columns].copy()
    new_subset = new_df[all_columns].copy()

    # Merge to find differences
    merged = pd.merge(
        original_subset,
        new_subset,
        on=key_columns_new,
        how='outer',
        suffixes=('_orig', '_new'),
        indicator=True
    )

    # Missing rows: present in original but not in new
    missing_from_new = merged[merged['_merge'] == 'left_only']

    # Compute differences
    comparison_cols = [col for col in all_columns if col not in key_columns_new]
    for col in comparison_cols:
        merged[f"{col}_diff"] = merged[f"{col}_new"] - merged[f"{col}_orig"]

    # Filter rows where any difference exceeds threshold
    diff_mask = merged['_merge'] == 'both'
    for col in comparison_cols:
        diff_mask &= merged[f"{col}_diff"].abs() > threshold

    differing_rows = merged[diff_mask]

    return missing_from_new, differing_rows

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

# join week lookup per channel 
def joining_allWeeks_perChannel(df, x_col, y_col, week_tester, fillna_cols):
    # Drop all columns except 'week_ending' and 'ig_user_id'
    reduced_df = df[[x_col, y_col]]
    
    # Pivot the dataframe to get columns per channel
    pivot_df = reduced_df.pivot_table(index=x_col, columns=y_col, aggfunc='size', fill_value=np.nan )
    
    # Reset the index to flatten the dataframe
    pivot_df = pivot_df.reset_index()
    
    # Merge with the lookup dataframe to ensure all weeks are included for each channel
    result_df = week_tester[[x_col]].merge(pivot_df, on=x_col, how='left')
    
    # Melt the dataframe back to the original format
    melted_df = pd.melt(result_df, id_vars=[x_col], var_name=y_col, value_name='count').drop(columns='count')
    melted_df.sample()
    
    # Join back with the rest of the data using a left join
    final_df = pd.merge(melted_df, df, on=[x_col, y_col], how='left')

    fill_dict = df[fillna_cols].drop_duplicates().set_index(y_col).to_dict()
    # Extract fill values for each column
    fill_values = {col: df.set_index(y_col)[col].to_dict() for col in fill_dict.keys()}
    
    # Apply fillna for each column
    for col, values in fill_values.items():
        final_df[col] = final_df[y_col].map(values).fillna(final_df[col])

    return final_df
    
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

################################ single platform calculations
def include_uk_decision(df, lookup):
    temp = df.merge(lookup[['Channel ID', 'Excluding UK']], on=['Channel ID'] , how='left')
    return temp[~((temp['PlaceID']=='UK') & (temp['Excluding UK']=='Yes'))]
        
def calculate_weekly(df, platform, gam_info):
    msno.matrix(df)
    weekly_df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    weekly_df['PlatformID'] = platform
    weekly_df['YearGAE'] = gam_info['YearGAE']
    return weekly_df

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

def calculate_annualy(df, platform, gam_info, aggregation_type='new'):
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
                    return group['Reach'].sum() / gam_info['number_of_weeks']
    
            return group['Reach'].sum() / gam_info['number_of_weeks']
    
        # Apply the logic to each ServiceID-PlaceID group
        year_avg = df.groupby(['ServiceID', 'PlaceID']).apply(compute_reach).reset_index(name='Reach')
    else: 
        print('calculating the average and not divding by 52')
        year_avg = df.groupby(['ServiceID', 'PlaceID', 'w/c'])['Reach'].sum().reset_index()
        year_avg = year_avg.groupby(['ServiceID', 'PlaceID'])['Reach'].mean().reset_index()
    
    # Combine world and year averages
    annual_df = pd.concat([world_avg, year_avg], ignore_index=True)
    annual_df['PlatformID'] = platform
    annual_df['YearGAE'] = gam_info['YearGAE']

    return annual_df

def summary_excel(weekly_df, business_part, platform, gam_info, aggregation_type='new'):
    path = "../data/singlePlatform/output/"
    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'w/c', 'Reach']
    weekly_df = calculate_weekly(weekly_df, platform, gam_info)[col_order]
    
    file_path = f"weekly/{gam_info['file_timeinfo']}_WEEKLY_{platform}_{business_part}byCountry.xlsx"
    weekly_df.to_excel(path+file_path, index=None)
    print(f"saved weekly file for {business_part} as:\n {file_path}") 

    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'Reach']
    annual_df = calculate_annualy(weekly_df, platform, gam_info, aggregation_type)[col_order]
    
    file_path = f"{gam_info['file_timeinfo']}_{platform}_{business_part}.xlsx"
    annual_df.to_excel(path+file_path, index=None)
    print(f"saved annual file for {business_part} as:\n {file_path}")
    
    return weekly_df, annual_df
    
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

def calculate_weekly_sumServices(df, serviceID, platform, gam_info):
    
    # temporary to explain discrepancy to minnie's values
    df_weekly = df.groupby(['PlaceID', 'w/c'])['Reach'].sum().reset_index()
    df_weekly['Reach'] = df_weekly['Reach']
    df_weekly['ServiceID'] = serviceID
    df_weekly['PlatformID'] = platform
    df_weekly['YearGAE'] = gam_info['YearGAE']
    
    path = "../data/singlePlatform/output/"
    file_path = f"weekly/{gam_info['file_timeinfo']}_WEEKLY_{platform}_{serviceID}byCountry.xlsx"
    
    col_order = ['YearGAE', 'ServiceID', 'PlatformID', 'PlaceID', 'w/c', 'Reach']
    df_weekly.to_excel(path+file_path, index=None)
    
    return df_weekly

def process_overlap(data, service1, service2, grouped_service, 
                    overlaps, overlap_type, overlap_service_id, platform, gam_info, path,
                    country_codes, pop_size_col):
    """
    overlap_service_id = which service ID contains the overlap factor!
    """
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
        pivot_df = sainsbury_formula(pivot_df, pop_size_col, 
                                      services, 'Reach')
        
    # Assign grouped service
    pivot_df['ServiceID'] = grouped_service
    
    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platform}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)
    
    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = calculate_weekly_sumServices(pivot_df, grouped_service, platform, gam_info)
    annual_df = calculate_annualy(data[grouped_service]['weekly'], platform, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platform}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df
    
    return pivot_df, annual_df

def process_overlap_v2(data, service1, service2, grouped_service,
                       overlaps, overlap_type, overlap_service_id, platform, gam_info, path,
                       country_codes, pop_size_col, service3=None, ):  
    # <-- Add service3 as an optional argument
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
        pivot_df = sainsbury_formula(pivot_df, pop_size_col, services, 'Reach')
            
    # Assign grouped service
    pivot_df['ServiceID'] = grouped_service
    
    # Export
    file_name = f"{gam_info['file_timeinfo']}_{platform}_{grouped_service}byCountry.xlsx"
    pivot_df.to_excel(f"../data/overlaps_datasets/{file_name}", index=None)
    
    # Weekly and annual aggregation
    data[grouped_service]['weekly'] = calculate_weekly_sumServices(pivot_df, grouped_service, platform, gam_info)
    annual_df = calculate_annualy(data[grouped_service]['weekly'], platform, gam_info)
    annual_file = f"{gam_info['file_timeinfo']}_{platform}_{grouped_service}.xlsx"
    annual_df.to_excel(path + annual_file, index=None)
    data[grouped_service]['annual'] = annual_df
    
    return pivot_df, annual_df


def calculate_aggregated_services(data, stages, platform, gam_info, path, 
                                  overlaps, country_codes, pop_size_col):

    for stage in stages:
        grouped_service, s1, s2, o_type, o_id, use_v2, s3 = stage
        data[grouped_service] = {'weekly': 'tbd', 'annual': 'tbd'}
        
        if use_v2:
            pivot, annual = process_overlap_v2(
                data=data,
                service1=s1,
                service2=s2,
                grouped_service=grouped_service,
                overlaps=overlaps,
                overlap_type=o_type,
                overlap_service_id=o_id,
                platform=platform,
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
                platform=platform,
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

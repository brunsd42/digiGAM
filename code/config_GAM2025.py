gam_info = {
    'lookup_file': 'GAM2025_Lookup.xlsx',
    'YearGAE': 2025,
    'file_timeinfo': 'GAM2025',
    
    'w/c_start': '2024-04-01', #first week that is included in this years GAM 
    'weekEnding_start': '2024-04-07', #first week that is included in this years GAM 
    'w/c_end': '2025-03-24', #last week that is included in this years GAM 
    'weekEnding_end': '2025-03-30', #last week that is included in this years GAM 
    'number_of_weeks': 52, # not every year has 52 weeks
    
    'overlap_viewer_uniqueViever': 1.1373,
    "DeviceFactor2020": 0.733520067685261,
    "DeviceFactor2020_nonWSL": 0.736660823133268,
    
    'business_units': {
        'WSL': {
            'Service IDs':['AFA', 'FRE', 'AMH', 'ARA', 'AZE', 'BEN', #'FOA',
                             'POR', 'BUR', 'MAN', 'DAR', 'FAR', 'KRW', 'GUJ',
                             'HAU', 'HIN', 'IGB', 'INO', 'KOR', 'KYR', 'ELT',
                             'MAR', 'SPA', 'NEP', 'PAS', 'PER', 'PDG', 'PUN',
                             'RUS', 'SER', 'SIN', 'SOM', 'SWA', 'TAM', 'TEL',
                             'THA', 'TIG', 'TUR', 'ECH', 'UKR', 'URD', 'UZB',
                             'VIE', 'YOR'],
            'exclude_UK': False,
            'sainsbury': {'TWI': False, 
                          'YT-': False,
                          'FBE': False,
                          'TTK': False,
                          'INS': False
                         }
        },    
        'GNL': {
            'Service IDs':['GNL'],
            # instead the lookup - Excluding UK, if Yes it will be excluded if No it will be included
            #'exclude_UK': False, 
            'sainsbury': {'TWI': False, 
                          'YT-': True,
                          'FBE': False,
                          'TTK': True,
                          'INS': False
                         }
        },    
        'WOR': {
            'Service IDs':['WOR'],
            'exclude_UK': True,
            'sainsbury': 
                        {'TWI': False,
                         'YT-': True,
                         'FBE': False,
                         'TTK': True,
                         'INS': False
                         }
        },    
        'WSE': {
            'Service IDs':['WSE'],
            'exclude_UK': False,
            'sainsbury': {'TWI': True,
                          'YT-': False,
                          'FBE': False,
                          'TTK': False,
                          'INS': False
                         }
        },    
        'MA-': {
            'Service IDs':['MA-'],
            'exclude_UK': False,
            'sainsbury': {'TWI': False, 
                          'YT-': False,
                          'FBE': False,
                          'TTK': True,
                          'INS': False
                         }
        },    
        'FOA': {
            'Service IDs':['FOA'],
            'exclude_UK': False,
            'sainsbury': {'TWI': False, 
                          'YT-': False,
                          'FBE': False,
                          'TTK': False,
                          'INS': False
                         }
        },
    },
    
    'ratings_file': '../../4 Ratings/Ratings_2025.xlsx', 
    'Source1': '',
    'Source2': '',
    'Source3': '',
    'Source4': '',
    'Notes': ''
}

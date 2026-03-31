# Site 

Ingestion 
---------

Ingesting data for site is coming from two sources. The majority of datapoints comes from Piano via API. The digi team has created a number of reports that gather reach figures on various platform and business levels, given the unique advantage that piano already reports **unique visitors** which we can directly use for our reach metric. 
Each API query that returns values is stored in a .csv file with the columns:
- Report No. 
- Unique Visitors
- geo_country

In addition there can be additional supporting columns that will be relevant in processing the datasets.

---

Manual ingestion for Site covers partner sources that do not report through Piano.
These datasets are supplied annually via Excel by BBC Learning English and BBC.com Syndication Partners (GNL).  
They provide reach values which we align to our weekly reach metric once expanded across all GAM weeks.

Each manual source file is read and stored as a .csv file with the columns:
- w/c  
- m_unique_visitors  
- PlaceID  
- ServiceID  
- PlatformID  


Processing
---------
Processing the automated data files is aimed to complete the gaps in PlatformID, ServiceID, PlaceID based on the API queries. Especially ServiceID needs to be completed from various sources including a mapping of the API query to the responding ServiceID, the Service mentioned in site_level2 in the API queries, and ultimately from fields such as language, NonJS or App. Finally all ServiceID values reported as “GNL” are normalised to the base service “BNO”.
Further the countries provided by Piano are mapped to PlaceID by piano's provided geo_country. 
Testing includes site, platform and country recognition and testing for any missing weeks per piano report. 

---

Processing the manual partner data ensures that identifiers supplied in the Excel files are aligned with the digiGAM schema. Country and platform fields are validated against the standard lookup tables, and all ServiceID values reported as “GNL” are normalised to the base service “BNO”. The dataset is then joined to the GAM week lookup to attach YearGAE and WeekNumber_finYear. Final tests confirm that each partner service has valid PlaceID, PlatformID and ServiceID entries and that all required GAM weeks are present.

Combining Sources
---------

Calculating Reach 
---------
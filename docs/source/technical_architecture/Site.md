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
There are a handful of partners that are not ingestable via Piano and therefore provide their data (currently) annually via excel sheets, these are: 
- Learning English Partners
- BBC.com Syndication Partners

For the annual **Global Audience Measurement** these manual sources are ingested in addition to provide a more complete picutre of Site reach.

Processing
---------
Processing the automated data files is aimed to complete the gaps in PlatformID, ServiceID, PlaceID based on the API queries. Especially ServiceID needs to be completed from various sources including a mapping of the API query to the responding ServiceID, the Service mentioned in site_level2 in the API queries, and ultimately from fields such as language, NonJS or App.
Further the countries provided by Piano are mapped to PlaceID by piano's provided geo_country. 
Testing includes site, platform and country recognition and testing for any missing weeks per piano report. 

---
Regarding the *manual* ...

Combining Sources
---------

Calculating Reach 
---------
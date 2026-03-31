# Business Overview

## What the pipeline delivers

The **digiGAM Calculation Pipeline** produces a weekly, business-wide estimate of **audience reach** based on **de-duplicated unique viewers** across all supported digital platforms, including:

- Web (Piano / AT Internet)
- YouTube
- Meta platforms (Facebook, Instagram)
- TikTok
- X/Twitter
- Podcasts
- Smaller platforms 
- Syndication partners

Combined, these feeds generate a harmonized view of how many people we reach each week, including:

- **Unique viewers**
- **Country-level audience splits**
- **Service- and platform-level rollups**

These outputs feed into:

- **Leadership** – weekly reach, content effectiveness, strategic prioritization  
- **Analytics** – dashboards, variance analysis, campaign performance  
- **Operations** – early detection of data issues and ingestion failures

A lookup sheet is used to combine all the variables that are used across the
pipeline, including the service and platform hierarchy, the social media
accounts the site-api queries and the country and time codes used. See the
[lookup sheet](https://docs.google.com/spreadsheets/d/1Z8tEC5OviUf_QwTbfseEE4i59rB63Yxm6WxCqIAGCLw/edit?usp=sharing)
for details.

---

## How the system works (high-level)

The pipeline standardizes heterogeneous platform data into a consistent structure.

### **Ingestion**
Each platform has a dedicated ingestion job that:

- Pulls raw performance & audience data from platform APIs or BBC Audiences Redshift
- Loads lookup tables (countries, services, platforms, weeks)  
- Runs automated structural tests:
  - Empty/missing reports  
  - Missing or duplicate keys  
  - Incomplete joins  
  - File size anomalies  

### **Processing & Enrichment**
Once raw files are ingested, each platform’s processing script can have these processing steps:

- Combine weekly CSVs (where applicable)
- Enrich ServiceID (helper dimensions) and PlaceID (rolling averages where missing)
- Calculate weekly visitors from the platform specific metrics
- Runs join integrity tests using a standardized test suite  
- Outputs a **processed table** for each platform

### **Business Unit Formation & Reach Calculation**
blub blub blub

### **Cross-Platform Calculation**
After all platforms are processed, digiGAM:

- Applies device and population scaling rules defined in the GAM window
- Performs cross-platform **deduplication** to avoid over-counting
- Produces weekly reach on all platform service and business unit level by country. 

---

## Key definitions

### **Reach (unique viewers)**
Estimated number of people exposed to BBC content in a given week, after removing duplicates **within** and **across** platforms.

### **Country split**
Distribution of audience by country.  
Derived from:

- Native platform geo metadata (Piano, Meta, TikTok where available)
- Modelled attribution rules when geo is missing

### **GAM window**
Each GAM Year (e.g., “GAM2026”) defines:

- Week-commencing range (`w/c_start → w/c_end`)
- The financial year begins in April

This ensures that all teams reference **the same weekly window and definitions**.

---

## Update cadence & delivery

- **Schedule:** Weekly (Thursday morning) or annually on the first Thursday in the new financial year
- **Automated:** Ingestion tests, processing enrichment, and logbook creation  
- **Outputs:**  
  - Platform-level processed tables  
  - Cross-platform unified reach tables  
  - Weekly full reach file (`.csv`)  
  - Supporting lookup consistency reports  

All datasets are fully test-logged and reproducible.

---

## Governance

- **Product owner:** Domi Bruns  
- **Technical ownership:** Data Engineering & Data Science  
- **Testing:** Automated ingestion/processing QA + daily logbook  
- **Traceability:** Raw → processed → combined → reach outputs  

The system ensures transparency and reproducibility of audience metrics used across BBC Commercial.

---

## Scope & limitations

- Covers all audience, excluding UK audiences that are reached by BBC public service. 
- Modelled country splits when no geographic metadata is provided
- Conservative deduplication ensures **under-counting is preferred over over-counting**
- Some platforms provide only aggregate signals; others provide user-level proxies  
- Pipeline structure is consistent, but platform schemas differ (handled through a unified enrichment layer)

---

## Where to learn more

- **FAQ** — common business and technical questions  
- **Technical Architecture** — ingestion flow, processing logic, dependencies  
- **API Reference** — function-level descriptions, platform mappings  
- **Glossary** — full definitions of all fields used in digiGAM  
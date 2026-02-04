
# Technical Architecture

This section describes the end-to-end flow, key modules, and data contracts.

## High-level flow
## Components

### Ingestion

---

#### Facebook

**Tables used**

- **Engagement:**  
  `central_insights.adverity_social_facebook_by_page`
- **Country distribution:**  
  `central_insights.adverity_social_facebook_page_fans_by_country`

**Weekly ingestion behaviour**

- **If engagement data is missing for a week**  
  → The week is **omitted entirely** for Facebook.  
  No fallback and no estimation is applied.

- **If country split is missing but engagement exists**  
  → A **historic rolling average country %** is applied for that page/service.

**Reasoning**

- Engagement totals cannot be imputed reliably.  
- Country percentages can be approximated from recent behaviour, especially because Facebook’s country distribution is based on **Followers**, which change gradually week to week.

#### YouTube

**Tables used**

- **Engagement:**  

- **Country distribution:**  
  `central_insights.adverity_social_facebook_page_fans_by_country`

**Weekly ingestion behaviour**

- **If engagement data is missing for a week**  
  → The week is **omitted entirely** for Facebook.  
  No fallback and no estimation is applied.

- **If country split is missing but engagement exists**  
  → A **historic rolling average country %** is applied for that page/service.

**Reasoning**

- Engagement totals cannot be imputed reliably.  
- Country percentages can be approximated from recent behaviour, especially because Facebook’s country distribution is based on **Followers**, which change gradually week to week.


#### Instagram

**Tables used**

- **Engagement:**  
  `central_insights.adverity_social_instagram_by_posts`
- **Country distribution:**  
  `central_insights.adverity_social_instagram_by_page_demo`

**Weekly ingestion behaviour**

- **If engagement data is missing for a week**  
  → The week is **omitted entirely** for Facebook.  
  No fallback and no estimation is applied.

- **If country split is missing but engagement exists**  
  → A **historic rolling average country %** is applied for that page/service.

**Reasoning**

- Engagement totals cannot be imputed reliably.  
- Country percentages can be approximated from recent behaviour, especially because Facebook’s country distribution is based on **Followers**, which change gradually week to week.


#### TikTok

**Data Source**

- Emplifi API

**Weekly ingestion behaviour**

**Reasoning**


#### Twitter

**Table used**

- **Engagement:**  
  `central_insights.adverity_social_instagram_by_posts`
- **Country distribution:**  
  

**Weekly ingestion behaviour**

**Reasoning**



### Transforms
- **Normalization**: Harmonize metrics and field names
- **Country attribution**: Apply modeled geo splits when missing
- Emit **curated** datasets with schemas suitable for de-duplication

### De-duplication
- Intra-platform: deterministic where IDs are available
- Cross-platform: modeled overlaps with device factors and guardrails
- Outputs **unique viewers** by platform/service/country/week

### Rollups
- Aggregate from **service** to **business unit** based on taxonomy
- Respect **exclude_UK** and platform-inclusion flags
- Produce weekly **business** reach

### Outputs & Monitoring
- Publish weekly reach tables/files for dashboards and analysts
- QC checks compare platform totals and attribution confidence
- Alerts on missing weeks/accounts, schema drift, or anomalous deltas

## Configuration (`gam_info`)
- **Year**: `YearGAE` and `file_timeinfo` (e.g., `GAM2026`)
- **Time windows**: `w/c_start` → `w/c_end`, `weekEnding_start` → `weekEnding_end`
- **Population/device factors**: used in overlap and normalization
- **Business units**: service lists, UK exclusion, platform flags

## Runbook (short)
- **Local build (docs)**: `cd docs && make html`
- **Pipeline orchestration**: (Airflow/Prefect/etc.) — add your DAG/job names here
- **Backfill**: Trigger date/partitioned runs as needed (link to internal runbook)

## Data contracts
Document field names, types, and primary keys for final outputs in `docs/data-models.md` (optional future page).
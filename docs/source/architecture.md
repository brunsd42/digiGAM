
# Technical Architecture

This section describes the end-to-end flow, key modules, and data contracts.

## High-level flow
## Components

### Ingestion

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

## More detailed documentation
```{toctree}
:maxdepth: 1

technical_architecture/Facebook
technical_architecture/Instagram
technical_architecture/Other Platforms
technical_architecture/Podcasts
technical_architecture/Site
technical_architecture/Social Media
technical_architecture/TikTok
technical_architecture/YouTube
``

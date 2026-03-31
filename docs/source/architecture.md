
# Technical Architecture

This section describes the end-to-end flow, key modules, and data contracts.

## High-level flow

- Ingestion of raw data 
- Processing to weekly reach metric
- Combine all platform-related sources into one file - reach x geography
- Calculate Business Units, aka Service Hierarchies
- Combine Social Media Platforms (WSC)
- Combine Total Digital (WT-)

## Design Principles

- Ingestion always creates the “raw weekly” data structure
- Platform duplication is a source characteristic, not processing logic
- Processing should never create new rows, only new columns
- Tests depend on ingestion having full week×platform coverage
- ServiceID is a semantic concept, not a source property

## Components

### Ingestion
TODO: add redshift background details 
TODO: add ws lookup table details

### Processing
- **Normalization**: Harmonize metrics and field names
- **Country attribution**: Apply modeled geo splits when missing
- **Base Services**: GNL gets remapped to BNO
- Emit **curated** datasets with schemas suitable for de-duplication
- Outputs **unique viewers** by platform/service/country/week

### Business Hiearchy Calculation
- Aggregate from **service** to **business unit** based on taxonomy
- Respect **exclude_UK** and platform-inclusion flags

### Platform Aggregation
- Cross-platform: modeled overlaps with device factors and guardrails

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

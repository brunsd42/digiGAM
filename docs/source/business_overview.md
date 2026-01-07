---

## Business Overview

**Path:** `docs/source/business_overview.md`

```markdown
# Business Overview

## What the pipeline delivers
The **digiGAM Calculation** pipeline produces a weekly **business-wide reach** metric based on **de-duplicated unique viewers** across multiple platforms (e.g., YouTube, Meta, TikTok, X, podcasts) and **country splits**.

Outputs serve:
- **Leadership & product**: performance tracking, strategy.
- **Analytics**: dashboards, variance analysis.
- **Ops**: early detection of data issues.

## Key definitions
- **Reach (unique viewers)**: Estimated unique audience after removing duplicates within and across platforms/services.
- **Country split**: Allocation of audience to countries, using native platform geo where available, otherwise a modeled attribution.
- **GAM window**: The **GAM year** configuration controls the reporting time frame (e.g., `GAM2026`), including:
  - Week-commencing window (`w/c_start` → `w/c_end`)
  - Week-ending window (`weekEnding_start` → `weekEnding_end`)
  - Population baselines and device factors
  - Business-unit rollups and platform inclusion flags

## Update cadence & delivery
- **Schedule**: Weekly (early Monday UTC, configurable)
- **Artifacts**: Curated tables (by service, country, business) and weekly business reach

## Governance
- **Product owner**: Domi Bruns  
- **Tech owners**: Data Engineering & DS  
- **Escalation**: (add your Slack/Teams channel or runbook link)

## Scope & limitations
- Focus on **owned/organic** audience unless explicitly stated
- Modeled country splits where native geo is not available
- Conservative cross-platform de-duplication to avoid over-counting

## Where to learn more
- FAQ — common questions
- Architecture — technical detail
- API Reference — functions, modules, and helpers
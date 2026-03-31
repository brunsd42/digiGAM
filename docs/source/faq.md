
# FAQ

## What deduplication is done 
- Device overlap
- Visitor vs visit metrics 
- Referrals for additional deduplication 

If no overlaps are available sainsbury is used to deduplicate reach. There are no overlaps calculated across countries. 

## How does de-duplicate across platforms work?
Provided overlap or sainsbury formula

## What if a data source fails for a week?
If a platform does not provide data for a given week, digiGAM will not create estimated reach or engagement for that platform.
Those metrics simply remain missing for that week.
However, if only the country split data is missing (but the engagement totals are present), digiGAM applies a fallback using a historic rolling average of previous country distributions.
👉 Technical details on platform‑specific fallbacks are provided in the Technical Architecture section.

--- 
When calculating the annual reach - missing weeks will be patched. Most commonly the rolling average of the previous four weeks will be used for the given week. 


## What is the GAM window?
DigiGAM is measured weekly starting from Mondays. The GAM year is the financial year, so starting in April. 

## Are country demographics reliably supplied and which platforms provide it?
Country demographics are available on a visitor base for Site. For YouTube and TikTok, they are provided on a content level and for Instagram and Facebook they are provided on a subscriber / follower level. X does not provide country demographics anymore and we now rely on historical country demographics from 2023. 

## Where are the weekly results?

## How are services rolled up to business units?

## Where do I see if the pipeline ran successfully?

## What are the most common issues when running the platform? 
- missing inputs from redshift

## What are the next developments for the digiGAM pipeline? 

## Why is GNL remapped to BNO in the processing?
Some platforms, such as YouTube and Instagram, report audience for BBC News at a more granular level, separating BBC News India (BNI) from BBC News Other (BNO). To keep all platforms aligned and ensure the same service hierarchy applies everywhere, we normalise GNL from all other sources to the base service BNO during processing. This avoids overlap between BNI and BNO and allows the business-unit rollup layer to rebuild GNL consistently based on its two base services.

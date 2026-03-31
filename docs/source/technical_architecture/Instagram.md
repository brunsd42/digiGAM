# Instagram

Ingestion
---------

**Engagement:**  
```
SELECT
  p.week_commencing,
  l.ig_account_id as ig_user_id,
  GREATEST(0, SUM(COALESCE(p.engagements_week_diff, p.engagements))) AS engagements,
  GREATEST(0, SUM(COALESCE(p.media_views_week_diff, p.video_views))) AS impressions,
  GREATEST(0, SUM(
      CASE
          WHEN UPPER(p.media_type) = 'VIDEO' THEN COALESCE(p.media_views_week_diff, p.video_views)
          ELSE 0
      END
  )) AS media_views,
  GREATEST(0, MAX(r.weekly_reach)) AS weekly_reach
FROM
  central_insights.adverity_social_instagram_by_posts AS p
RIGHT JOIN
      world_service_audiences_insights.social_media_lookup_ig AS l
  ON 
      p.account_id = l.ig_account_id
LEFT JOIN
      central_insights.adverity_social_instagram_by_reach AS r
  ON 
      p.week_commencing = r.week_commencing
  AND 
      p.account_id = r.account_id
WHERE
      p.week_commencing IS NOT NULL
  AND
      p.week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
GROUP BY
  p.week_commencing,
  l.ig_account_id
  ;
```
This query retrieves weekly Instagram performance per account by joining post‑level data with the GAM Instagram lookup and optional reach data.
For each (account_id, week), it aggregates engagements, impressions, and video views (using week‑diff fields where available) and takes the maximum reported weekly reach.
The result is one row per week per account containing the key engagement and reach metrics needed for GAM processing.

---

**Country distribution:** 
```
SELECT 
  week_commencing,
  l.ig_account_id as page_id,
  page_name, 
  country_code,
  country_name as ins_country_name,
  followers_by_demographic
FROM
  redshiftdb.central_insights.adverity_social_instagram_by_page_demo AS p
RIGHT JOIN
      world_service_audiences_insights.social_media_lookup_ig AS l
  ON 
      p.page_id = l.ig_identifier
WHERE
      week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
  AND
      followers_by_demographic > 0
  AND 
      country_name <> ''
  ;
```

This query returns weekly Instagram follower counts by country.
It RIGHT JOINs the demographic table to the GAM lookup (redshift version) so all tracked accounts appear, even if Adverity has gaps.

- Filters to the GAM reporting window (w/c_start → w/c_end)
- Includes only rows with valid country info and follower counts > 0
- Output = one row per (week, page_id, country) with follower totals and page metadata.

Processing
---------
**Engagements**: 
Until 10 November 2025, Instagram engagement and reach metrics were unreliable, so GAM uses values from an alternative “stale” data source for all weeks before this date. From 10 November 2025 onwards, the pipeline switches to the new query‑based metrics.

In the next step weekly Instagram view data is joined with the reels replay factor, a per‑account multiplier used to convert video views into estimated plays. **static file wuold need updating when new accounts come up (but because data delivery it's useless) or add global average to fill new accounts wilth**
Missing replay factors are replaced with the overall mean to ensure complete coverage across all channels. Core metrics (engagements, media_views, impressions, weekly_reach) are normalised by filling missing values with zero.
Estimated plays are calculated as:
```
plays = media_views / reels_replay_factor
```
These plays are then adjusted with service‑specific multipliers to create the "30 view" metric, which captures platform differences in Reels performance.
An IG Modelled Factor (add more detail about def safe_ratio(row):) is applied to convert engagements and impressions into a modelled estimate of engaged reach. **Persian (PER) accounts** use a dedicated model variant reflecting different audience behaviour.
The final engaged_reach value is computed using engagements and adjusted plays, with a special override for the Persian News account. All engaged reach values are capped at the platform‑reported weekly reach to prevent modelled inflation.

**Country**:
Country percentages are calculated after removing unknown locations (PlaceID = 'UNK'). For each (Channel ID, w/c), follower counts are summed across all remaining countries, and each country’s share is then computed as followers_by_demographic / weekly_total. The code validates that these country percentages sum to 1 for every account-week combination.
Country-based internet shutdowns this year meant we had to manually set the country percentage to a fraction of it's reach. specifically This year was Iran, for two weeks in January and since early March. 
Missing weeks are filled using a four‑week rolling average before calculating percentages. (done in 3_ script)

Combining Sources & Calculating Reach 
---------
Engagement data is first merged with country‑percentage data; rows that match directly are kept, while unmatched rows fall back to annual average country splits. These two sets are then combined into a unified table containing country‑level engaged reach for each account and week. After merging with country metadata and removing duplicates, missing engaged‑reach and percentage values are filled with zero, and final uv_by_country values are computed as:
```
uv_by_country = engaged_reach × country_%
```

This produces clean, country‑level weekly Instagram engagement estimates aligned with GAM’s country taxonomy. This outputs the columns: 
```
['w/c', 'PlaceID', 'ServiceID', 'Channel ID', 'uv_by_country']
```

Account Details
---------------
accounts on Redshift:
these accounts are pretty inactive: 
- 17841413658250442
- 17841405021536770
- 17841402244137767

lost access:
- 17841435042594492

```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_active_accounts.csv
```

accounts that are not on Redshift:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_inactive_accounts.csv
```

unclear accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_unclear_accounts.csv
```

Cadence
-------
- **If engagement data is missing for a week**  
  → The week is **omitted entirely** for Facebook.  
  No fallback and no estimation is applied.

- **If country split is missing but engagement exists**  
  → A **historic rolling average country %** is applied for that page/service.

**Reasoning**

- Engagement totals cannot be imputed reliably.  
- Country percentages can be approximated from recent behaviour, especially because Facebook’s country distribution is based on **Followers**, which change gradually week to week.

Useful tests
../test/issue_lists_2026-03-19/INS_1e_06_issue_list.csv
INS_1e_07_issue_list
action: INS_1e_10_issue_list add to weekly report if any accounts had an outlier 
        in week 2-3-26 should have raised alarms bells_ INS_1e_11


add tests: 
1e add missing weeks per service AFTER stale data has added
3: missing services / weeks 
3: unique entries in engagmenets and countries to prevent 1:many joins
add a test to see if cap is successfully applied 

outlier convo once all data is in 
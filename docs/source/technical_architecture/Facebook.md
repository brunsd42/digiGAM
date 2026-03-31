# Facebook

Ingestion
---------

### source query

**Engagement:**  
```SELECT
      week_commencing,
      page_id,
      CASE
          WHEN (AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user) 
                > AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer)
          THEN ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user)) 
                + (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer))*0.04827
          ELSE (AVG(video_views)/AVG(page_video_views_to_10s_unique_viewer)) 
                + ((AVG(engagements)/AVG(post_engagements_to_page_consumptions))/AVG(avg_engagements_per_user))*0.04822
      END AS engaged_reach
  FROM 
      redshiftdb.central_insights.adverity_social_facebook_by_page AS p
  RIGHT JOIN
      world_service_audiences_insights.social_media_lookup_fb AS l
      ON p.page_id = l.fb_page_id
  WHERE 
      period = 'week'
  AND
      week_commencing BETWEEN '{gam_info['w/c_start']}' AND '{gam_info['w/c_end']}'
  GROUP BY 
      week_commencing, page_id
      ;
```
      
Facebook performance data is joined to a lookup table to align each row with the correct Facebook page. For each page–week pair, two normalised metrics are calculated:

```
E = AVG(engagements)
    / AVG(post_engagements_to_page_consumptions)
    / AVG(avg_engagements_per_user)
```

Video-View Ratio:
```
V = AVG(video_views)
    / AVG(page_video_views_to_10s_unique_viewer)
```

The query then compares `E` and `V`:

- If `E > V`:
```
engaged_reach = E + (V * 0.04827)
```
- Otherwise:
```
engaged_reach = V + (E * 0.04822)
```

This produces a blended `engaged_reach` value for each page and week, where the dominant behaviour (engagement or video viewing) forms the main signal and the secondary behaviour contributes a small weighted adjustment.

**Country distribution:**  
```SELECT 
      week_commencing,
      page_id ,
      page_name,
      page_fans_country_total AS page_fans_country,
      country_code AS fb_metric_breakdown
  FROM
      redshiftdb.central_insights.adverity_social_facebook_page_fans_by_country
  WHERE
      week_commencing Between '{gam_info['w/c_start']}' and '{gam_info['w/c_end']}'
      ;
```
It selects weekly Facebook page‑fan data by joining each record to its page ID, page name, and country breakdown. The query filters rows to only include weeks within the specified range defined by gam_info['w/c_start'] and gam_info['w/c_end'], which can be found in the helper/config.py. For every valid week and page, it returns the total number of fans in that country alongside the relevant country code.


**Weekly ingestion behaviour**

- **If engagement data is missing for a week**  
  → The week is **omitted entirely** for Facebook.  
  No fallback and no estimation is applied.

- **If country split is missing but engagement exists**  
  → A **historic rolling average country %** is applied for that page/service.

**Reasoning**

- Engagement totals cannot be imputed reliably.  
- Country percentages can be approximated from recent behaviour, especially because Facebook’s country distribution is based on **Fans**, which change gradually week to week.

Processing
---------
**Engagements**: engagements are merged with the lookup of socialmedia accounts. That sheet also contains start and end dates of the accounts. Given that Redshift returns 0 for the weeks before an account was launched, we remove these weeks to not falsely lower the annual average of the account. 

**Country**
Country percentage is calculating once all the country gaps are filled, **unknown are excluded (currently not!)** 
if one platform removes it we shohld tdo the same for others
to calculate a global number and then the percentage of every country. Missings weeks are filled with the average of four weeks. 
Country-based internet shutdowns this year meant we had to manually set the country percentage to a fraction of it's reach. specifically This year was Iran, for two weeks in January and since early March. 

Combining Sources & Calculating Reach 
---------


Account Details
---------------
active accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/FBE_active_accounts.csv
```


inactive accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/FBE_inactive_accounts.csv
```

unclear accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/FBE_unclear_accounts.csv
```

not measured:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/FBE_not measured_accounts.csv
```
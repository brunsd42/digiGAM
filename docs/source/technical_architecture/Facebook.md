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


Ingestion
=========
Channel List 
100163990128209
100826406677543
101242425117794
10150118096995434
102186576502930
102775241582303
103678496456574
103894041188535
1048672825173961
1068750829805728
108618735892959
109993787372855
110602040605523
113097918700687
114577681901041
1218471521614509
124158667615790
125958540790873
126548377386804
127031120644257
127284984048667
127616880628817
105750467635355
129550730451583
1296222730434431
1385243528221504
133536249999517
143048895744759
146214266618
151173098230967
1514652022126019
147105585468223
1526071940947174
159142754459988
160817274042538
160894643929209
1633841096923106
163565280410742
1648071085436082
166580710064489
167959249906191
171824429536304
173190249410689
173825495996416
151955124848859
179118645433239
163571453661989
1801602293398610
1842714285954391
186742265162
190992343324
192168680794107
207150596007088
215963631764
228458913833525
228571877602966
230299653821
1799887850269003
236659822607
237647452933504
260669183761
26363622695
264572343581678
282681245570
285361880228
286182898229953
298318986864908
303095973088848
303522857815
312215209342
314780205644442
228735667216
331621733578582
341294666718736
347501767628
359687864111179
367167334474
388094534609616
408902742628123
422868177848193
463402774003391
485274381864409
490671421264757
510203778998227
526813830804091
545688452250873
592266607456680
630866223444617
64040652712
654070648098812
660673490805047
81395234664
818434098337843
232455204224
935986439746020
938609046278894
9432520138
237388053065908
279678448760878
317278538359186
832942320102956
948946275170651

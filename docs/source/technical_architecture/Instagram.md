# Instagram
## Ingestion
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


Account Details
---------------
active accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_active_accounts.csv
```


inactive accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_inactive_accounts.csv
```

unclear accounts:
```{csv-table}
:header-rows: 1
:file: ../_static/tables/INS_unclear_accounts.csv
```

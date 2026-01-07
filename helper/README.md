
# helper/

Shared utilities for the reach pipeline.

- `config.py` — non-secret configuration (e.g., `gam_info`).
- `functions.py` — **general-purpose helpers** used across scripts (I/O, schema checks, date utilities, safe operations, lightweight transforms).
- `test_functions.py` — **data-quality test helpers** used *by* tests across the repo to assert pipeline health (e.g., no missing weeks, accounts present, schema contracts).
- `security_config.py` — secret loaders (reads env vars; does not store secrets).

> Rule of thumb: if it’s reusable across multiple scripts or tests and not business-rule-heavy, it probably belongs here.

---

## Quick start

```python
# In your scripts
from helper.functions import lookup_loader, sainsbury_formula
from helper.test_functions import test_weeks_presence_per_account, test_lookup_files
``

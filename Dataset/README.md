# Dataset

All data in this folder belong to the current Y-Map implementation.

| Path | Contents |
|---|---|
| `YMap/Object_Models/` | 11 JSON object models (classes, attributes, associations, inheritance) |
| `YMap/Generated/candidates.jsonl` | The 728 generated candidate schemas with their mapping decisions and logical metadata |
| `YMap/Generated/generation_manifest.json` | Generator backend, per-system counts, timings and environment |
| `YMap/All_SQL_Solutions/<system>/SQL_Solutions/<schema_id>.sql` | Canonical DDL for each candidate |

Candidates per object model (`python` backend):

| Object model | Candidates |
|---|---|
| banking | 8 |
| blog_platform | 8 |
| crm | 16 |
| customer_order | 4 |
| ecommerce | 24 |
| erp_system | 384 |
| healthcare | 12 |
| hospital_information_system | 192 |
| library | 16 |
| travel_booking | 16 |
| university | 48 |
| **Total** | **728** |

Generating candidates does not measure performance. Measurements and evaluation outputs are under `results/YMap/`; see the top-level `README.md` and `STATUS.md`.

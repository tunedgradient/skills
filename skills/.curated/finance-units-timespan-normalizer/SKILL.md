---
name: "finance-units-timespan-normalizer"
description: "Normalize finance tables across sources by standardizing unit scale, currency labels, and reporting-window labels into canonical long-form CSV output. Use when importing statements, KPIs, or data exports into models, comps, or variance reports; not for full modeling or forecasting."
---

# Finance Units + Reporting-Window Normalizer

Normalize messy finance tables into a consistent, machine-friendly CSV.

## Run the normalizer

```bash
python scripts/normalize_units_timespans.py --input <path> --output normalized_financials.csv --notes normalization_notes.md
```

Optional flags:

- `--sheet <name>`: Read a specific sheet for XLSX input.
- `--entity <ticker_or_name>`: Fill `entity` on every output row.
- `--source <tag>`: Override source label in output rows.

## Acceptable inputs

- CSV or XLSX exports from vendors, IR sites, BI tools, or ERPs.
- Long-form tables (`metric`, `value`, `window` style columns).
- Wide-form tables (window columns such as `Q1 2025`, `FY 2024`, `LTM`, `YTD 2025`, `9M 2025`).

## Guaranteed outputs

- `normalized_financials.csv`
- `normalization_notes.md`

Output schema:

- `entity`
- `metric_raw`
- `metric`
- `value_raw`
- `value`
- `unit`
- `scale`
- `window_label`
- `window_end_date`
- `window_start_date`
- `as_of_date`
- `source`
- `notes`

## Operating rules

- Keep currency as reported in v1; do not perform FX conversion.
- Detect scale from hints such as `in thousands`, `000s`, `mm`, `million`, `bn`, `billion`.
- Detect currency from symbols/codes (`$`, `USD`, `EUR`, `GBP`, `JPY`).
- Normalize reporting windows to canonical labels when possible (`Q1 2025`, `FY 2024`, `LTM`, `YTD 2025`, `9M 2025`).
- Reshape wide tables to one metric-window observation per row.
- Parse parentheses negatives and percent values (`35%` -> `0.35`).
- Keep percent and per-share metrics out of currency table-scale multiplication.
- Record assumptions and ambiguities in `normalization_notes.md`.

## Reference docs

- `references/time-span-conventions.md`
- `references/unit-scale-currency.md`

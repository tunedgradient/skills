# Time-Span Conventions

Use these canonical labels in `window_label`:

- Quarter: `Q1 2025`, `Q2 2025`, `Q3 2025`, `Q4 2025`
- Fiscal year: `FY 2024`
- Rolling window: `LTM` or `TTM` (preserve source label if both are valid)
- Partial year: `9M 2025`, `6M 2025`, `3M 2025`, `1H 2025`, `2H 2025`
- Year to date: `YTD 2025`

## Ambiguity handling

- If the source only says `Q1`, `YTD`, or `9M` with no year:
  - Preserve the label fragment.
  - Add a row note (for example `window_year_unknown`).
- If fiscal year-end month is unknown:
  - Preserve `FY` labels.
  - Avoid deriving fiscal start/end dates.
- If the window is unparseable:
  - Keep the original string.
  - Add a row note (for example `window_unparsed`).

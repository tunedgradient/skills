# Unit Scale and Currency Cues

## Scale keywords to multiplier

- `in thousands`, `000s` -> x1,000
- `in millions`, `mm`, `million` -> x1,000,000
- `in billions`, `bn`, `billion` -> x1,000,000,000

## Currency cues

- `$`, `USD`, `US$` -> `USD`
- `€`, `EUR` -> `EUR`
- `£`, `GBP` -> `GBP`
- `¥`, `JPY` -> `JPY`

## Exceptions and guardrails

- EPS/per-share metrics:
  - Treat as per-share values.
  - Do not apply table-scale multipliers (`000`, `mm`, `bn`).
- Percent metrics:
  - Parse as decimal (`35%` -> `0.35`).
  - Avoid currency scaling.
- Ambiguous currency:
  - Leave `unit` blank.
  - Record ambiguity in row notes and `normalization_notes.md`.

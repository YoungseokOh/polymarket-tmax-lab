# Market Rules

## Supported Rule Families

### Wunderground airport templates
- Seoul / Incheon Intl Airport Station
- NYC / LaGuardia Airport Station
- London / London City Airport Station
- Core fields parsed:
  - official source URL
  - airport station name/id
  - target local date
  - Celsius or Fahrenheit
  - whole-degree language
  - finalization and revision policy

### Hong Kong Observatory Daily Extract
- Supported via `CLMMAXT`
- Core fields parsed:
  - official station code, such as `HKA`
  - HKO source URL
  - Celsius binning
  - finalized-data language

### Central Weather Administration
- Supported via official station/CODiS wording
- Core fields parsed:
  - station id, such as `466920`
  - official source family
  - Celsius binning
  - finalization language

## Outcome Families
- integer Celsius labels such as `8°C`
- Fahrenheit two-degree ranges such as `70-71°F`
- lower catch-alls such as `8°C or below`
- upper catch-alls such as `50°F or higher`

## Resolution Behavior
- Markets wait for finalized source data.
- Revisions after finalization are ignored when the rule text says so.
- The bin mapper expands labels to settlement intervals using the market precision rule:
  - `8°C` -> `[7.5, 8.5)`
  - `70-71°F` -> `[34.0? no]` not generic conversion; it becomes the rule-specific two-degree Fahrenheit settlement interval centered on the labeled integer endpoints.

## Examples
- Seoul: Wunderground / Incheon Intl Airport / Celsius / whole degree
- NYC: Wunderground / LaGuardia / Fahrenheit / two-degree labeled ranges
- London: Wunderground / London City Airport / Celsius / whole degree
- Hong Kong: HKO `CLMMAXT` / HKA / Celsius / official daily extract
- Taipei: CWA station `466920` / Celsius / official station observations

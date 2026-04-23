# Market Rules

## Use This When
- `scan-markets`
- `rule_parser.py`, `market_filter.py`, `market_spec.py`
- truth source adapters
- settlement fidelity 검토

## Core Rules
- generic city weather로 대체하지 않는다
- official source, station, local date, unit, outcome bin, finalization 문구를 유지한다
- source adapter가 불완전하면 fail closed 한다

## v1 Supported Families
- Seoul / Wunderground / Incheon Intl Airport
- NYC / Wunderground / LaGuardia Airport
- London / Wunderground / London City Airport
- Hong Kong / Hong Kong Observatory
- Taipei / Central Weather Administration

## First Reads
- `docs/markets/market-rules.md`
- `docs/codebase/markets.md`
- `docs/codebase/weather.md`

## Change Sync
- 규칙 파싱이 바뀌면 parser tests와 `docs/markets/market-rules.md`를 같이 갱신한다
- official source handling이 바뀌면 truth adapter 문서와 테스트를 같이 갱신한다

from pmtmax.markets.outcome_schema import parse_outcome_label, parse_outcome_schema


def test_parse_outcome_label_accepts_unicode_dash_and_spacing_variants() -> None:
    parsed = parse_outcome_label("50–51 °F")

    assert parsed.lower == 50
    assert parsed.upper == 51


def test_parse_outcome_label_accepts_upper_and_lower_aliases() -> None:
    upper = parse_outcome_label("49°F or above")
    lower = parse_outcome_label("35°F or lower")

    assert upper.lower == 49
    assert upper.upper is None
    assert lower.lower is None
    assert lower.upper == 35


def test_parse_outcome_label_accepts_open_interval_bounds() -> None:
    parsed = parse_outcome_label("<29°F")

    assert parsed.upper == 29
    assert parsed.upper_inclusive is False


def test_parse_outcome_schema_infers_missing_unit_from_peer_labels() -> None:
    parsed = parse_outcome_schema(["53–54°F", "55–56°F", "57 or higher"])

    assert parsed[-1].lower == 57
    assert parsed[-1].upper is None

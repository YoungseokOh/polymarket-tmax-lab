"""Outcome label parsing and normalization."""

from __future__ import annotations

import re
from typing import Literal

from pmtmax.markets.market_spec import OutcomeBin

_DASH_TRANSLATION = str.maketrans({
    "–": "-",
    "—": "-",
    "−": "-",
})
_UNIT_RE = re.compile(r"(?P<unit>[CF])\b", re.I)
LOWER_RE = re.compile(
    r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*or\s*(?:below|lower|under)\s*$",
    re.I,
)
UPPER_RE = re.compile(
    r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*or\s*(?:higher|above|over)\s*$",
    re.I,
)
LOWER_BOUND_RE = re.compile(r"^\s*(?:<|<=|≤)\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*$", re.I)
UPPER_BOUND_RE = re.compile(r"^\s*(?:>|>=|≥)\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*$", re.I)
EXACT_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*$", re.I)
RANGE_RE = re.compile(
    r"^\s*(?P<low>-?\d+(?:\.\d+)?)\s*-\s*(?P<high>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[CF])?\s*$",
    re.I,
)


def infer_unit_from_label(label: str) -> Literal["C", "F"]:
    """Infer the temperature unit from an outcome label."""

    normalized = _normalize_label(label)
    if match := _UNIT_RE.search(normalized):
        return match.group("unit").upper()  # type: ignore[return-value]
    return "C"


def _normalize_label(label: str) -> str:
    normalized = label.translate(_DASH_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _resolved_unit(match: re.Match[str], default_unit: Literal["C", "F"] | None) -> Literal["C", "F"] | None:
    unit = match.groupdict().get("unit")
    if unit:
        return unit.upper()  # type: ignore[return-value]
    return default_unit


def infer_unit_from_labels(labels: list[str]) -> Literal["C", "F"]:
    """Infer the common unit across outcome labels, falling back to Celsius."""

    inferred_units = {infer_unit_from_label(label) for label in labels if _UNIT_RE.search(_normalize_label(label))}
    if len(inferred_units) == 1:
        return next(iter(inferred_units))
    return "C"


def parse_outcome_label(label: str, default_unit: Literal["C", "F"] | None = None) -> OutcomeBin:
    """Parse Polymarket outcome text into a numeric interval."""

    stripped = label.strip()
    normalized = _normalize_label(stripped)

    if match := LOWER_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, upper=value)

    if match := UPPER_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, lower=value)

    if match := LOWER_BOUND_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, upper=value, upper_inclusive=False)

    if match := UPPER_BOUND_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, lower=value, lower_inclusive=False)

    if match := RANGE_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        low = float(match.group("low"))
        high = float(match.group("high"))
        return OutcomeBin(label=stripped, lower=low, upper=high)

    if match := EXACT_RE.match(normalized):
        if _resolved_unit(match, default_unit) is None:
            msg = f"Unsupported outcome label: {label}"
            raise ValueError(msg)
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, lower=value, upper=value)

    msg = f"Unsupported outcome label: {label}"
    raise ValueError(msg)


def parse_outcome_schema(labels: list[str]) -> list[OutcomeBin]:
    """Parse a sequence of market outcome labels."""

    default_unit = infer_unit_from_labels(labels)
    return [parse_outcome_label(label, default_unit=default_unit) for label in labels]

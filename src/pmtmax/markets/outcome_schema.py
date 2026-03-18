"""Outcome label parsing and normalization."""

from __future__ import annotations

import re
from typing import Literal

from pmtmax.markets.market_spec import OutcomeBin

LOWER_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?(?P<unit>[CF])\s*or\s*below\s*$", re.I)
UPPER_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?(?P<unit>[CF])\s*or\s*higher\s*$", re.I)
EXACT_RE = re.compile(r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*°?(?P<unit>[CF])\s*$", re.I)
RANGE_RE = re.compile(
    r"^\s*(?P<low>-?\d+(?:\.\d+)?)\s*-\s*(?P<high>-?\d+(?:\.\d+)?)\s*°?(?P<unit>[CF])\s*$",
    re.I,
)


def infer_unit_from_label(label: str) -> Literal["C", "F"]:
    """Infer the temperature unit from an outcome label."""

    if "f" in label.lower():
        return "F"
    return "C"


def parse_outcome_label(label: str) -> OutcomeBin:
    """Parse Polymarket outcome text into a numeric interval."""

    stripped = label.strip()

    if match := LOWER_RE.match(stripped):
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, upper=value)

    if match := UPPER_RE.match(stripped):
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, lower=value)

    if match := RANGE_RE.match(stripped):
        low = float(match.group("low"))
        high = float(match.group("high"))
        return OutcomeBin(label=stripped, lower=low, upper=high)

    if match := EXACT_RE.match(stripped):
        value = float(match.group("value"))
        return OutcomeBin(label=stripped, lower=value, upper=value)

    msg = f"Unsupported outcome label: {label}"
    raise ValueError(msg)


def parse_outcome_schema(labels: list[str]) -> list[OutcomeBin]:
    """Parse a sequence of market outcome labels."""

    return [parse_outcome_label(label) for label in labels]


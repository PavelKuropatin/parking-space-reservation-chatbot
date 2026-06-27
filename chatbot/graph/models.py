from datetime import datetime
import re
from typing import Callable, Optional

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    required: bool
    prompt: str
    validate: Callable[[str, dict], Optional[str]]


_PLATE_RE = re.compile(r"^[A-Z]{0,3}[-\s]?\d{2,4}[A-Z]{0,3}$", re.IGNORECASE)


def __validate_license_plate(value: str, _current_details: dict) -> Optional[str]:
    if not _PLATE_RE.match(value.strip()):
        return f"'{value}' doesn't look like a valid plate."
    return None


def __parse_dt(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def __validate_start_time(value: str, _current_details: dict) -> Optional[str]:
    dt = __parse_dt(value)
    if dt is None:
        return "I couldn't read that start time."
    if dt < datetime.now():
        return "That start time is in the past."
    return None


def __validate_end_time(value: str, current_details: dict) -> Optional[str]:
    dt = __parse_dt(value)
    if dt is None:
        return "I couldn't read that end time."
    start = __parse_dt(current_details.get("start_time") or "")
    if start and dt <= start:
        return "The end time must be after the start time."
    return None


RESERVATION_DETAILS_SPECS: list[FieldSpec] = [
    FieldSpec("customer_name", True, "What is your full name?", lambda v, s: None),
    # TODO how to validate space type correctly
    FieldSpec(
        "space_type",
        True,
        "Which parking space type do you want? (Standard, EV, Compact, Oversized, Accessible)",
        lambda v, s: None,
    ),
    FieldSpec("start_time", True, "When should the reservation start?", __validate_start_time),
    FieldSpec("end_time", True, "And when should it end?", __validate_end_time),
    FieldSpec("license_plate", True, "What's your licence plate?", __validate_license_plate),
]

RESERVATION_DETAILS_SPEC_BY_NAME = {s.name: s for s in RESERVATION_DETAILS_SPECS}


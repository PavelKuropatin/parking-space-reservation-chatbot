from __future__ import annotations

from decimal import Decimal
import json
from datetime import date, datetime, time
from typing import Any

from langchain_core.tools import BaseTool, tool

from chatbot.database.sql_store import ParkingData
from chatbot.settings import get_settings


def make_sql_store_tools(db: ParkingData) -> list[BaseTool]:
    """Build the parking tool set, each tool sharing the given ``db`` pool."""

    @tool(parse_docstring=True)
    def list_parking_locations(include_inactive: bool = False) -> str:
        """List the parking facilities, with their codes, city and capacity.

        Call this to discover the valid location codes used by the other tools.

        Args:
            include_inactive: If true, also return facilities that are not
                currently operational. Defaults to false.
        """
        return _dumps(db.get_locations(active_only=not include_inactive))

    @tool(parse_docstring=True)
    def list_space_types() -> str:
        """List the bookable space-type categories and their codes.

        Call this to discover the valid space-type codes (e.g. STANDARD, EV)
        used by the pricing and availability tools.
        """
        return _dumps(db.get_space_types(active_only=True))

    @tool(parse_docstring=True)
    def get_parking_prices(
        location_code: str,
        space_type_code: str | None = None,
    ) -> str:
        """Get the current per-hour and daily-max prices for a facility.

        Returns the active price list (VAT-inclusive) for the given location,
        optionally narrowed to a single space type.

        Args:
            location_code: Facility code, one of: CP-CENTRAL, CP-AIRPORT,
                CP-OLDTOWN. Use list_parking_locations if unsure.
            space_type_code: Optional space type to filter by, one of:
                STANDARD, COMPACT, OVERSIZED, EV, ACCESSIBLE. Omit for all types.
        """
        return _dumps(
            db.get_current_pricing(_norm(location_code), _norm(space_type_code))
        )

    @tool(parse_docstring=True)
    def get_price_history(
        location_code: str,
        space_type_code: str | None = None,
    ) -> str:
        """Get the full price history (current and past rates) for a facility.

        Use this only when the user asks about previous or upcoming rate
        changes; for the price a customer pays now, use get_parking_prices.

        Args:
            location_code: Facility code, one of: CP-CENTRAL, CP-AIRPORT,
                CP-OLDTOWN.
            space_type_code: Optional space type to filter by, one of:
                STANDARD, COMPACT, OVERSIZED, EV, ACCESSIBLE. Omit for all types.
        """
        return _dumps(
            db.get_pricing_history(_norm(location_code), _norm(space_type_code))
        )

    @tool(parse_docstring=True)
    def get_parking_availability(
        location_code: str,
        space_type_code: str | None = None,
    ) -> str:
        """Get the live count of free parking spaces at a facility right now.

        Returns total, available and occupied spaces per space type, plus the
        timestamp of the last sensor refresh.

        Args:
            location_code: Facility code, one of: CP-CENTRAL, CP-AIRPORT,
                CP-OLDTOWN.
            space_type_code: Optional space type to filter by, one of:
                STANDARD, COMPACT, OVERSIZED, EV, ACCESSIBLE. Omit for all types.
        """
        return _dumps(db.get_availability(_norm(location_code), _norm(space_type_code)))

    @tool(parse_docstring=True)
    def get_weekly_hours(location_code: str) -> str:
        """Get a facility's regular weekly opening hours (the normal schedule).

        Returns one row per ISO weekday (1=Monday .. 7=Sunday) with open/close
        times and the is_24h / is_closed flags. For a specific calendar date
        that may have a holiday override, use get_opening_hours_on_date instead.

        Args:
            location_code: Facility code, one of: CP-CENTRAL, CP-AIRPORT,
                CP-OLDTOWN.
        """
        return _dumps(db.get_working_hours(_norm(location_code)))

    @tool(parse_docstring=True)
    def get_opening_hours_on_date(
        location_code: str,
        on_date: str | None = None,
    ) -> str:
        """Get a facility's effective opening hours for one specific date.

        Holiday / event / maintenance overrides take precedence over the normal
        weekly schedule for that date. Always check is_closed and is_24h before
        reading opens_at / closes_at. Use this to answer "are you open on X?".

        Args:
            location_code: Facility code, one of: CP-CENTRAL, CP-AIRPORT,
                CP-OLDTOWN.
            on_date: The date to check, as ISO YYYY-MM-DD. Omit for today.
        """
        try:
            d = _parse_date(on_date)
        except ValueError as exc:
            return _dumps({"error": str(exc)})
        result = db.get_effective_hours(_norm(location_code), d)
        if result is None:
            return _dumps({"error": f"Unknown location_code '{location_code}'."})
        return _dumps(result)

    @tool(parse_docstring=True)
    def get_upcoming_closures(
        location_code: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        """List date-specific closures and special hours (holidays, maintenance).

        Returns overrides to the normal schedule within an optional date window.
        Useful for "any holiday closures coming up?" type questions.

        Args:
            location_code: Optional facility code to filter by (CP-CENTRAL,
                CP-AIRPORT, CP-OLDTOWN). Omit for all facilities.
            from_date: Optional start of the window, ISO YYYY-MM-DD (inclusive).
            to_date: Optional end of the window, ISO YYYY-MM-DD (inclusive).
        """
        try:
            start = _parse_date(from_date) if from_date else None
            end = _parse_date(to_date) if to_date else None
        except ValueError as exc:
            return _dumps({"error": str(exc)})
        return _dumps(db.get_special_hours(_norm(location_code), start, end))

    return [
        list_parking_locations,
        list_space_types,
        get_parking_prices,
        get_price_history,
        get_parking_availability,
        get_weekly_hours,
        get_opening_hours_on_date,
        get_upcoming_closures,
    ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert DB types (Decimal/date/time/datetime) to JSON-safe.

    Decimal -> float, date/time/datetime -> ISO-8601 string. Use on query
    results before json.dumps() or before returning from an agent tool.
    """
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, time, datetime)):
        return value.isoformat()
    return value


def _dumps(obj: Any) -> str:
    """JSON-encode a query result, coercing DB types to JSON-safe primitives."""
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)


def _norm(code: str | None) -> str | None:
    """Normalize a user/LLM-supplied code: trim + upper-case, None passes through."""
    return code.strip().upper() if code else None


def _parse_date(value: str | None) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string; None -> today. Raises ValueError."""
    if value is None or not value.strip():
        return date.today()
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{value}'. Use ISO format YYYY-MM-DD (e.g. 2026-06-24)."
        ) from exc

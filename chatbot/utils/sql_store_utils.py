from decimal import Decimal
import json
from datetime import date, datetime, time
from typing import Any

from langchain_core.tools import BaseTool, tool

from chatbot.database.sql_store import get_parking_data_db


def make_sql_store_tools() -> list[BaseTool]:
    """Build the parking tool set, each tool sharing the given ``db`` pool."""
    db = get_parking_data_db()



    @tool(parse_docstring=True)
    def get_parking_prices() -> str:
        """Get the current per-hour and daily-max prices.

        Returns the active price list (VAT-inclusive).
        """
        return _dumps(db.get_current_pricing())

    @tool(parse_docstring=True)
    def get_working_hours() -> str:
        """
        Get an actual working hours for parking
        Returns one row per ISO weekday (1=Monday .. 7=Sunday) 
        with open/close times and the is_24h / is_closed flags
        is_24h - flag if it is opened all day
        is_closed - flag if it is temporarly closed
        """
        # TODO humanize response fields?
        return _dumps(db.get_working_hours())

    return [
        get_parking_prices,
        get_working_hours,
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

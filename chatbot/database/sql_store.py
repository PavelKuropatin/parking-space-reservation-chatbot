from __future__ import annotations

import os
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from chatbot.settings import Settings


class ParkingData:
    """Query API over the CityPark dynamic DB, backed by a connection pool."""

    # -- lifecycle ----------------------------------------------------------

    def __init__(self, settings: Settings) -> None:

        url = "postgresql://{user}:{pswd}@{host}/{db}".format(
            user=settings.postgres_user, 
            pswd=settings.postgres_pswd,
            host=settings.postgres_host,
            db=settings.postgres_db,
        )

        self._pool = ConnectionPool(
            url,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )


    def open(self, *, wait: bool = False, timeout: float = 30.0) -> None:
        self._pool.open(wait=wait, timeout=timeout)


    def close(self) -> None:
        self._pool.close()


    def __enter__(self):
        return self


    def __exit__(self, *exc: object) -> None:
        self.close()


    def __fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def __fetch_one(self, query: str, params: dict | None = None) -> dict | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    # ======================================================================
    # locations
    # ======================================================================


    def get_locations(self, active_only: bool = True) -> list[dict]:
        """All parking facilities, ordered by name."""
        sql = """
            SELECT location_id, location_code, name, city, timezone,
                   total_spaces, is_active
            FROM   locations
            WHERE  (%(active_only)s::boolean IS FALSE OR is_active)
            ORDER  BY name;
        """
        return self.__fetch_all(sql, {"active_only": active_only})

    def get_location(self, location_code: str) -> dict | None:
        """A single facility by its stable code (e.g. 'CP-CENTRAL')."""
        sql = """
            SELECT location_id, location_code, name, city, timezone,
                   total_spaces, is_active
            FROM   locations
            WHERE  location_code = %s;
        """
        return self.__fetch_one(sql, (location_code,))

    # ======================================================================
    # space_types
    # ======================================================================

    def get_space_types(self, active_only: bool = True) -> list[dict]:
        """The catalogue of bookable space categories."""
        sql = """
            SELECT space_type_id, code, name, description, is_active
            FROM   space_types
            WHERE  (%(active_only)s::boolean IS FALSE OR is_active)
            ORDER  BY space_type_id;
        """
        return self.__fetch_all(sql, {"active_only": active_only})

    def get_space_type(self, code: str) -> dict | None:
        """A single space type by its code (STANDARD, COMPACT, EV, ...)."""
        sql = """
            SELECT space_type_id, code, name, description, is_active
            FROM   space_types
            WHERE  code = %s;
        """
        return self.__fetch_one(sql, (code,))

    # ======================================================================
    # pricing
    # ======================================================================

    def get_current_pricing(
        self,
        location_code: str | None = None,
        space_type_code: str | None = None,
    ) -> list[dict]:
        """Currently-active prices, optionally filtered.

        Joins in the location and space-type names/codes so the rows are
        self-describing. Pass either/both filters to narrow the result.
        """
        sql = """
            SELECT p.price_id,
                   l.location_code, l.name        AS location_name,
                   st.code         AS space_type_code,
                   st.name         AS space_type_name,
                   p.hourly_rate, p.daily_max_rate, p.currency, p.vat_rate
            FROM   pricing p
            JOIN   locations   l  ON l.location_id   = p.location_id
            JOIN   space_types st ON st.space_type_id = p.space_type_id
            WHERE  (%(loc)s::text  IS NULL OR l.location_code = %(loc)s)
              AND  (%(type)s::text IS NULL OR st.code         = %(type)s)
            ORDER  BY l.location_code, st.space_type_id;
        """
        return self.__fetch_all(sql, {"loc": location_code, "type": space_type_code})

    # ======================================================================
    # space_availability
    # ======================================================================

    def get_availability(
        self,
        location_code: str | None = None,
        space_type_code: str | None = None,
    ) -> list[dict]:
        """Current free-space snapshot per location + type, with names attached."""
        sql = """
            SELECT a.availability_id,
                   l.location_code, l.name AS location_name,
                   st.code AS space_type_code, st.name AS space_type_name,
                   a.total_spaces, a.available_spaces,
                   (a.total_spaces - a.available_spaces) AS occupied_spaces
            FROM   space_availability a
            JOIN   locations   l  ON l.location_id   = a.location_id
            JOIN   space_types st ON st.space_type_id = a.space_type_id
            WHERE  (%(loc)s::text  IS NULL OR l.location_code = %(loc)s)
              AND  (%(type)s::text IS NULL OR st.code         = %(type)s)
            ORDER  BY l.location_code, st.space_type_id;
        """
        return self.__fetch_all(sql, {"loc": location_code, "type": space_type_code})

    # ======================================================================
    # working_hours
    # ======================================================================

    def get_working_hours(
        self,
        location_code: str,
        day_of_week: int | None = None,
    ) -> list[dict]:
        """Regular weekly hours for a facility.

        ``day_of_week`` is ISO-8601 (1=Monday .. 7=Sunday); omit for all 7 rows.
        """
        sql = """
            SELECT oh.hours_id, l.location_code,
                   oh.day_of_week, oh.opens_at, oh.closes_at,
                   oh.is_24h, oh.is_closed
            FROM   working_hours oh
            JOIN   locations l ON l.location_id = oh.location_id
            WHERE  l.location_code = %(loc)s
              AND  (%(dow)s::int IS NULL OR oh.day_of_week = %(dow)s)
            ORDER  BY oh.day_of_week;
        """
        return self.__fetch_all(sql, {"loc": location_code, "dow": day_of_week})

    # ======================================================================
    # special_hours
    # ======================================================================

    def get_special_hours(
        self,
        location_code: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[dict]:
        """Date-specific overrides (holidays, events, maintenance).

        Optionally bound to a facility and/or a ``[from_date, to_date]`` window.
        """
        sql = """
            SELECT sh.special_id, l.location_code,
                   sh.calendar_date, sh.opens_at, sh.closes_at,
                   sh.is_closed, sh.description
            FROM   special_hours sh
            JOIN   locations l ON l.location_id = sh.location_id
            WHERE  (%(loc)s::text  IS NULL OR l.location_code = %(loc)s)
              AND  (%(from)s::date IS NULL OR sh.calendar_date >= %(from)s)
              AND  (%(to)s::date   IS NULL OR sh.calendar_date <= %(to)s)
            ORDER  BY l.location_code, sh.calendar_date;
        """
        return self.__fetch_all(
            sql, {"loc": location_code, "from": from_date, "to": to_date}
        )

    # ======================================================================
    # Composite / "effective" queries — the runtime tools
    # ======================================================================

    def get_price_list(self, location_code: str) -> list[dict]:
        """Current price list for a facility (active prices only)."""
        return self.get_current_pricing(location_code=location_code)

    def get_live_availability(self, location_code: str) -> list[dict]:
        """Live availability for one facility (alias for get_availability by code)."""
        return self.get_availability(location_code=location_code)

    def get_effective_hours(
        self,
        location_code: str,
        on_date: date | None = None,
    ) -> dict | None:
        """Effective opening hours for a facility on a given date.

        A ``special_hours`` override wins over the weekly ``working_hours``
        rule for that calendar date (COALESCE precedence). Treat ``is_closed``
        and ``is_24h`` as authoritative before reading the times. Returns one
        row, or ``None`` if the location code is unknown.
        """
        sql = """
            SELECT l.location_code,
                   l.name AS location_name,
                   %(d)s::date                            AS for_date,
                   EXTRACT(ISODOW FROM %(d)s::date)::int  AS day_of_week,
                   COALESCE(sh.is_closed, wh.is_closed)   AS is_closed,
                   wh.is_24h,
                   COALESCE(sh.opens_at,  wh.opens_at)    AS opens_at,
                   COALESCE(sh.closes_at, wh.closes_at)   AS closes_at,
                   sh.description                         AS override_reason
            FROM   locations l
            JOIN   working_hours wh
                   ON wh.location_id = l.location_id
                  AND wh.day_of_week = EXTRACT(ISODOW FROM %(d)s::date)
            LEFT   JOIN special_hours sh
                   ON sh.location_id   = l.location_id
                  AND sh.calendar_date = %(d)s::date
            WHERE  l.location_code = %(loc)s;
        """
        return self.__fetch_one(
            sql, {"loc": location_code, "d": on_date or date.today()}
        )

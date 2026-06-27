from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from chatbot.settings import get_settings
from chatbot.utils.singleton import Singleton


class ParkingDataDB(metaclass=Singleton):
    """Query API over the CityPark dynamic DB, backed by a connection pool."""


    def __init__(self) -> None:
        settings = get_settings()

        url = "postgresql://{user}:{pswd}@{host}/{db}".format(  # pylint: disable=consider-using-f-string
            user=settings.postgres_user,
            pswd=settings.postgres_pswd,
            host=settings.postgres_host,
            db=settings.postgres_db,
        )

        self._pool = ConnectionPool(
            url,
            min_size=settings.postgres_pool_min_size,
            max_size=settings.postgres_pool_max_size,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=True,
        )

    def open(self, *, wait: bool = False, timeout: float = 30.0) -> None:
        self._pool.open(wait=wait, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self._pool.close()
        self.close()

    def __fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # ======================================================================
    # pricing
    # ======================================================================
    def get_current_pricing(self) -> list[dict]:
        """Currently-active prices."""
        sql = """
            SELECT p.price_id,
                   p.price_type,
                   p.amount,
                   p.currency,
                   p.description 
            FROM   pricing p
        """
        return self.__fetch_all(sql)

    # ======================================================================
    # working_hours
    # ======================================================================
    def get_working_hours(self) -> list[dict]:
        """Regular weekly hours for a facility."""
        sql = """
            SELECT wh.hours_id,
                   wh.day_of_week, 
                   wh.opens_at, wh.closes_at,
                   wh.is_24h, wh.is_closed
            FROM   working_hours wh
            ORDER BY wh.day_of_week;
        """
        return self.__fetch_all(sql)

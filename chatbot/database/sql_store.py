from cachetools import TTLCache, cached

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from chatbot.settings import get_settings

_RESPONSE_CACHE = TTLCache(maxsize=100, ttl=60)


class ParkingDataDB:

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

    def __fetch_all(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        with self._pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    def __write(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        with self._pool.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)

                result = cursor.fetchall() if cursor.description else None

                connection.commit()

                return result

    def get_space_pricing(self) -> list[dict]:
        sql = """
            SELECT p.price_id,
                   p.price_type,
                   p.amount,
                   p.currency,
                   p.description 
            FROM pricing p;
        """
        return self.__fetch_all(sql)

    def get_working_hours(self) -> list[dict]:
        sql = """
            SELECT v.hours_id,
                   v.day_of_week, 
                   v.opens_at,
                   v.closes_at,
                   v.temporaly_closed
            FROM  vw_working_hours v
            ORDER BY v.day_of_week;
        """
        return self.__fetch_all(sql)


    @cached(_RESPONSE_CACHE)
    def get_avaliable_spaces(self) -> list[dict]:
        sql = """
            SELECT v.level,
                   v.space_type,
                   v.available_spaces
            FROM vw_avaliable_spaces v;
        """
        return self.__fetch_all(sql)

    def write_reservation(self, params: dict) -> list[dict]:
        sql = """
            INSERT INTO reservations (level, space_type, customer_full_name, license_plate, start_datetime, end_datetime) 
            VALUES (%(level)s, %(space_type)s, %(customer_full_name)s, %(license_plate)s, %(start_datetime)s, %(end_datetime)s) 
            RETURNING id
        """
        return self.__write(sql, params)

    def udpate_reservation_status(self, params: dict) -> list[dict]:
        sql = """
            UPDATE reservations
            SET status = %(reservation_status)s
            WHERE id = %(reservation_id)s
            RETURNING id
        """
        return self.__write(sql, params)


__PARKING_DATA_DB: ParkingDataDB = None


def get_parking_data_db() -> ParkingDataDB:
    global __PARKING_DATA_DB
    if __PARKING_DATA_DB is None:
        __PARKING_DATA_DB = ParkingDataDB()
    return __PARKING_DATA_DB

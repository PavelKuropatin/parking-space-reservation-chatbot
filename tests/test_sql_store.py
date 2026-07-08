from unittest.mock import patch

import chatbot.database.sql_store as store


def static_db() -> store.ParkingDataDB:
    with patch.object(store, "ConnectionPool"):
        return store.ParkingDataDB.__new__(store.ParkingDataDB)


def test_pricing_and_working_hours_do_not_share_a_cache_entry():
    db = static_db()
    with patch.object(
        db, "_ParkingDataDB__fetch_all", side_effect=[["PRICING_ROW"], ["HOURS_ROW"]]
    ) as mock_fetch_all:
        pricing = db.get_space_pricing()
        hours = db.get_working_hours()

    assert pricing == ["PRICING_ROW"]
    assert hours == ["HOURS_ROW"]
    assert mock_fetch_all.call_count == 2


def test_repeated_calls_are_served_from_cache():
    db = static_db()
    with patch.object(
        db, "_ParkingDataDB__fetch_all", return_value=["ROW"]
    ) as mock_fetch_all:
        db.get_space_pricing()
        db.get_space_pricing()
        db.get_working_hours()
        db.get_working_hours()

    assert mock_fetch_all.call_count == 2

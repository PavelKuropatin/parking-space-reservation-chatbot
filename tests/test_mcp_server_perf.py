import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chatbot.mcp.mcp_server import ReservationMcpModel, submit_reservation

CONCURRENT_REQUESTS_N = 50


def static_token():
    return SimpleNamespace(scopes=["reservations:submit"], client_id="admin-agent")


def static_reservation(**overrides):
    fields = dict(
        customer_full_name="John Smith",
        level="B1",
        space_type="EV",
        start_datetime="2026-07-04 09:00",
        end_datetime="2026-07-04 19:00",
        license_plate="LV-1234",
        approved_ts="20260703T160000",
    )
    fields.update(overrides)
    return ReservationMcpModel(**fields)


@pytest.fixture(autouse=True)
def reservations_paths(tmp_path, monkeypatch):
    reservations_file = tmp_path / "reservations.psv"
    lock_file = tmp_path / "reservations.lock"
    monkeypatch.setattr("chatbot.mcp.mcp_server.RESERVATIONS_FILE", reservations_file)
    monkeypatch.setattr("chatbot.mcp.mcp_server.RESERVATIONS_LOCK_FILE", lock_file)
    return reservations_file


# --------------------------------------------------------------------------- #
# Concurrency: the read-count-then-append sequence is only safe because of the
# fcntl lock in submit_reservation. This guards against the duplicate-id race
# the code comment refers to (caught under load previously).
# --------------------------------------------------------------------------- #
def test_submit_reservation_assigns_unique_ids_under_concurrent_load(reservations_paths):
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=static_token()):
        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS_N) as pool:
            results = list(
                pool.map(
                    lambda i: submit_reservation(static_reservation(license_plate=f"LV-{i}")),
                    range(CONCURRENT_REQUESTS_N),
                )
            )

    assert all(r["is_error"] is False for r in results)
    ids = [r["reservation_id"] for r in results]
    assert sorted(ids) == list(range(1, CONCURRENT_REQUESTS_N + 1))

    lines = reservations_paths.read_text(encoding="utf-8").splitlines()
    assert len(lines) == CONCURRENT_REQUESTS_N
    recorded_ids = sorted(int(line.split(" | ", 1)[0]) for line in lines)
    assert recorded_ids == list(range(1, CONCURRENT_REQUESTS_N + 1))

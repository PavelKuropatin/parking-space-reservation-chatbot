from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chatbot.mcp.mcp_server import ReservationMcpModel, has_required_scope, submit_reservation


def static_token(scopes=("reservations:submit",), client_id="admin-agent"):
    return SimpleNamespace(scopes=list(scopes), client_id=client_id)


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


# has_required_scope
def test_has_required_scope_returns_token_when_scope_present():
    token = static_token(scopes=["reservations:submit"])
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=token):
        result = has_required_scope("reservations:submit")

    assert result is token


def test_has_required_scope_raises_when_scope_missing():
    token = static_token(scopes=["other:scope"])
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=token):
        with pytest.raises(PermissionError):
            has_required_scope("reservations:submit")


# submit_reservation
def test_submit_reservation_writes_first_record(reservations_paths):
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=static_token()):
        result = submit_reservation(static_reservation())

    assert result == {"is_error": False, "reservation_id": 1}
    content = reservations_paths.read_text(encoding="utf-8")
    assert content == (
        "1 | John Smith | LV-1234 | 2026-07-04 09:00 : 2026-07-04 19:00 | B1-EV | 20260703T160000\n"
    )


def test_submit_reservation_increments_id_and_appends(reservations_paths):
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=static_token()):
        submit_reservation(static_reservation())
        result = submit_reservation(static_reservation(customer_full_name="Jane Doe"))

    assert result == {"is_error": False, "reservation_id": 2}
    lines = reservations_paths.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("2 | Jane Doe")


def test_submit_reservation_blocks_without_required_scope(reservations_paths):
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=static_token(scopes=[])):
        result = submit_reservation(static_reservation())

    assert result == {
        "is_error": True,
        "error_message": "Unsufficient permissions",
        "error_http_code": 401,
    }
    assert not reservations_paths.exists()


def test_submit_reservation_returns_500_on_unexpected_error(reservations_paths):
    with patch("chatbot.mcp.mcp_server.get_access_token", return_value=static_token()), patch(
        "builtins.open", side_effect=OSError("disk full")
    ):
        result = submit_reservation(static_reservation())

    assert result == {
        "is_error": True,
        "error_message": "disk full",
        "error_http_code": 500,
    }

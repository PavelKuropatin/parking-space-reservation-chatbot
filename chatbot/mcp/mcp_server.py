import os
from pathlib import Path

from pydantic import BaseModel, Field

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.dependencies import get_access_token
from chatbot.logging import logger

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)
RESERVATIONS_FILE = DATA_DIR / "reservations.psv"


class ReservationMcpModel(BaseModel):
    customer_full_name: str = Field(
        description="Customer full name. e.g. 'Pavel Kuropatin'"
    )
    level: str = Field(description="Parking level name, e.g. 'B1'")
    space_type: str = Field(description="Parking space type, e.g. 'EV'")
    start_datetime: str = Field(
        description="Reservation start datetime, YYYY-MM-DD HH:MM format, e.g. '2026-07-04 09:00'"
    )
    end_datetime: str = Field(
        description="Reservation end datetime, YYYY-MM-DD HH:MM format, e.g. '2026-07-04 19:00'"
    )
    license_plate: str = Field(
        description="Vehicle number/license plate, e.g. 'LV-1234'"
    )
    approved_ts: str = Field(
        description="Reservation approval datetime, ISO format, e.g. '20260703T160000'"
    )


verifier = StaticTokenVerifier(
    tokens={
        "admin-agent-token": {
            "client_id": "admin-agent",
            "scopes": ["reservations:submit"],
        },
    },
    required_scopes=["reservations:submit"],
)

mcp = FastMCP("parking", auth=verifier)


def has_required_scope(scope: str):
    token = get_access_token()
    if scope not in (token.scopes or []):
        raise PermissionError(f"Token of '{token.client_id}' lacks '{scope}' scope")
    return token


@mcp.tool
def submit_reservation(reservation: ReservationMcpModel) -> dict:
    """Submit a reservation. Returns reservation_id."""
    try:
        _ = has_required_scope("reservations:submit")

        # TODO lock?
        if os.path.exists(RESERVATIONS_FILE):
            with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
                reservation_id = f.read().count("\n") + 1

        else:
            reservation_id = 1

        record = " | ".join(
            [
                str(reservation_id),
                reservation.customer_full_name,
                reservation.license_plate,
                f"{reservation.start_datetime} : {reservation.end_datetime}",
                f"{reservation.level}-{reservation.space_type}",
                reservation.approved_ts,
            ]
        )

        with open(RESERVATIONS_FILE, "a+", encoding="utf-8") as f:
            f.write(record + "\n")

        return {"is_error": False, "reservation_id": reservation_id}
    except PermissionError as e:
        logger.error("Unsufficient permissions: %s", e)
        return {"is_error": True, "error_message": "Unsufficient permissions", "error_http_code": 401}
    except Exception as e:
        logger.error("Failed to submit_reservation: %s", e)
        return {"is_error": True, "error_message": str(e), "error_http_code": 500}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8088, path="/mcp")

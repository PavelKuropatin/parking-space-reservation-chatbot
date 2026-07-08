from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, Field, model_validator

from langchain_core.messages import AnyMessage

RESERVATION_FIELD_LABELS = {
    "customer_full_name": "Customer",
    "level": "Parking level",
    "space_type": "Space type",
    "start_datetime": "Start time",
    "end_datetime": "End time",
    "license_plate": "Plate",
}


class UserInputType(BaseModel):
    """Root routing decision for a fresh turn."""

    route: Literal["information", "reservation", "_unknown"] = Field(
        description=(
            "'reservation' - if the user is providing/confirming/changing reservation details or otherwise driving the booking; use it if user directly write about it. "
            "'information' - a question about prices, facilities, policies, locations, EV charging, opening hours, etc. "
            "'unknown' - message is unclear, greetings or cannot be classified "
            "If the message provides or changes ANY reservation detail, choose 'reservation'"
        )
    )


class ReservationUpdates(BaseModel):

    customer_full_name: Optional[str] = Field(
        None, description="Customer/user full name."
    )
    level: Optional[str] = Field(
        None, description="Parking level name (eg. B1, B2 or B3)."
    )
    space_type: Optional[str] = Field(
        None, description="Parking place type (STANDARD, EV or OVERSIZED)"
    )
    start_datetime: Optional[datetime] = Field(
        None, description="Reservation start datetime in YYYY-MM-DD HH:MM format without TZ."
    )
    end_datetime: Optional[datetime] = Field(
        None,
        description="Reservation end datetime in YYYY-MM-DD HH:MM format without TZ. Must be after start.",
    )
    license_plate: Optional[str] = Field(
        None, description="Vehicle license plate exactly as stated, e.g. 'AB-1234'."
    )
    intent: Literal["provide", "confirm", "modify", "other"] = Field(
        "provide",
        description=(
            "Intent of the user's LAST message. "
            "'confirm' = approves the shown summary as-is; "
            "'modify' = wants to change an already-given value; "
            "'provide' = supplying/refining details; "
            "'other' = greeting/question/etc."
        ),
    )


class Reservation(BaseModel):

    customer_full_name: str = Field(..., description="Customer/user full name.")
    level: str = Field(..., description="Parking level name (eg. B1, B2 or B3).")
    space_type: str = Field(
        ..., description="Parking place type (STANDARD, EV or OVERSIZED)"
    )
    start_datetime: datetime = Field(
        ..., description="Reservation start datetime in YYYY-MM-DD HH:MM format  without TZ."
    )
    end_datetime: datetime = Field(
        ...,
        description="Reservation end datetime in YYYY-MM-DD HH:MM format  without TZ. Must be after start.",
    )
    license_plate: str = Field(
        ..., description="Vehicle license plate exactly as stated, e.g. 'AB-1234'."
    )

    @model_validator(mode="after")
    def _order(self):
        if self.end_datetime <= self.start_datetime:
            raise ValueError("end_datetime must be after start_datetime")
        return self


class ReservationPhase(Enum):
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    REGUESTING_APROVAL = "request_admin_approval"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReservationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    reservation_messages: Annotated[list[AnyMessage], add_messages]
    qa_messages: Annotated[list[AnyMessage], add_messages]
    data_recording_messages: Annotated[list[AnyMessage], add_messages]

    route: Literal["reservation", "information", "_unknown"]

    # guardrail
    input_blocked: bool
    block_reason: str

    # control checkpointer messages
    human_message: str

    # reservation
    intent: str
    reservation_details: dict
    reservation_issues: list[str]
    changed_this_turn: bool
    reservation: Reservation
    reservation_phase: ReservationPhase
    reservation_id: int
    reservation_status: ReservationStatus

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage


class UserIntentDecision(BaseModel):
    """Root routing decision for a fresh turn."""

    route: Literal["information_request", "reservation", "unknown"] = Field(
        description=(
            "'reservation' - user wants to book, reserve, or change a parking spot; use it if user directly write about it "
            "'information_request' - a question about hours, pricing, rules, and another questions about."
            "'unknown' - message is unclear, greetings or cannot be classified"
        )
    )


RESERVATION_FIELD_LABELS = {
    "customer_name": "Customer",
    "space_type": "Space type",
    "start_time": "Start time",
    "end_time": "End time",
    "license_plate": "Plate",
}


class ParkingReservationDetails(BaseModel):
    """Reservation fields. Leave a field null if the user did not mention it."""

    customer_name: Optional[str] = Field(
        None, description="Reservation customer user full name"
    )
    space_type: Optional[str] = Field(None, description="Parking space type")
    start_time: Optional[str] = Field(
        None,
        description="Reservation start datetime in YYYY-mm-DD HH:mm format, resolved against NOW",
    )
    end_time: Optional[str] = Field(
        None,
        description="Reservation end datetime YYYY-mm-DD HH:mm format, resolved against NOW",
    )
    license_plate: Optional[str] = Field(
        None, description="Vehicle plate as written, e.g. 'LV-1234' or 'AB1234'"
    )


class UserConfirmDecision(BaseModel):
    """User response interpretation to the confirmation prompt."""

    intent: Literal["yes", "cancel", "change"] = Field(
        description=(
            "'yes' - approve and book "
            "'cancel' - decline/cancel entirely "
            "'change' - the message contains new or corrected booking details."
        )
    )


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Optional[str]  # information_request | reservation | unknown

    # guardrail
    input_blocked: bool
    block_reason: str
    # control checkpointer messages
    human_message: str
    ai_message: str

    # information
    rag_context: list[str]
    db_context: list[dict]

    # reservation
    current_details: dict
    errors: dict
    pending_field: Optional[str]
    pending_error: Optional[str]
    intent: Optional[str]  # confirmation intent: yes | cancel | change
    reservation_phase: str  # "collecting" | "confirming" | "done" | "cancelled"

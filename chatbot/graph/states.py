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
    "license_plate": "Plate"
}

RESERVATION_FIELD_DESCRIPTIONS = {
    "customer_name": "customer/user first and last names",
    "space_type": "parking place type (STANDARD, EV or OVERSIZED)",
    "start_time": "reservation start datetime in YYYY-MM-DD HH:MM format",
    "end_time": "reservation end datetime in YYYY-MM-DD HH:MM format",
    "license_plate": "customer vehicle number / license plate",
}


class UserConfirmDecision(BaseModel):
    """User response interpretation to the confirmation prompt."""

    decision: Literal["yes", "cancel", "change"] = Field(
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

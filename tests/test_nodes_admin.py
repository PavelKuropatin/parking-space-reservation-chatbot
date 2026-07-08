import json
from datetime import datetime
from unittest.mock import patch

import pytest
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import GraphInterrupt

from chatbot.nodes import (
    after_request_admin_approval_router,
    process_admin_rejection_node,
    request_admin_approval_node,
    submit_reservation_admin_node,
)
from chatbot.states import Reservation, ReservationPhase, ReservationStatus


def static_reservation(**overrides):
    fields = dict(
        customer_full_name="John Smith",
        level="B1",
        space_type="EV",
        start_datetime=datetime(2026, 7, 4, 9, 0),
        end_datetime=datetime(2026, 7, 4, 19, 0),
        license_plate="LV-1234",
    )
    fields.update(overrides)
    return Reservation(**fields)


CONFIG = {"configurable": {"thread_id": "thread-1"}}


def static_tool_message(payload: dict, tool_call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(
        content=[{"type": "text", "text": json.dumps(payload)}],
        tool_call_id=tool_call_id,
    )


# request_admin_approval_node
def test_request_admin_approval_node_sends_interrupt_payload():
    state = {"reservation": static_reservation()}
    with patch(
        "chatbot.nodes.interrupt",
        return_value={"reservation_status": "approved", "approval_ts": "20260704T190500"},
    ) as mock_interrupt:
        request_admin_approval_node(state, CONFIG)

    payload = mock_interrupt.call_args[0][0]
    assert payload["calling_thread_id"] == "thread-1"
    assert payload["request_id"].startswith("admin-request-")
    assert "John Smith" in payload["message"]


def test_request_admin_approval_node_handles_approval():
    state = {"reservation": static_reservation()}
    with patch(
        "chatbot.nodes.interrupt",
        return_value={"reservation_status": "APPROVED", "approval_ts": "20260704T190500"},
    ):
        result = request_admin_approval_node(state, CONFIG)

    assert result["reservation_status"] == ReservationStatus.APPROVED
    assert result["reservation_details"]["approved_ts"] == "20260704T190500"
    assert result["reservation_details"]["customer_full_name"] == "John Smith"
    recorded = json.loads(result["data_recording_messages"].content)
    assert recorded == result["reservation_details"]


def test_request_admin_approval_node_handles_rejection():
    state = {"reservation": static_reservation()}
    with patch(
        "chatbot.nodes.interrupt",
        return_value={"reservation_status": "Rejected", "approval_ts": "20260704T190500"},
    ):
        result = request_admin_approval_node(state, CONFIG)

    assert result["reservation_status"] == ReservationStatus.REJECTED


def test_request_admin_approval_node_reraises_graph_interrupt():
    state = {"reservation": static_reservation()}
    with patch("chatbot.nodes.interrupt", side_effect=GraphInterrupt()):
        with pytest.raises(GraphInterrupt):
            request_admin_approval_node(state, CONFIG)


def test_request_admin_approval_node_recovers_from_bad_response():
    state = {"reservation": static_reservation()}
    with patch("chatbot.nodes.interrupt", return_value={"approval_ts": "x"}):
        result = request_admin_approval_node(state, CONFIG)

    assert result == {
        "route": None,
        "reservation_phase": None,
        "reservation_details": {},
        "reservation": None,
        "messages": [
            AIMessage("Failed to request admin approval. Please try again later")
        ],
    }


# process_admin_rejection_node
def test_process_admin_rejection_node():
    result = process_admin_rejection_node({})

    assert result == {
        "route": None,
        "reservation_phase": ReservationPhase.COMPLETED,
        "reservation_details": {},
        "reservation": None,
        "messages": [AIMessage("Your reservaton was rejected")],
    }


# submit_reservation_admin_node
def test_submit_reservation_admin_node_succeeds():
    tool_message = static_tool_message({"is_error": False, "reservation_id": 5})
    state = {
        "data_recording_messages": [HumanMessage("submit please"), tool_message],
        "reservation_messages": [AIMessage("confirmed")],
    }

    result = submit_reservation_admin_node(state)

    assert result["reservation_id"] == 5
    assert result["reservation_phase"] == ReservationPhase.COMPLETED
    assert result["reservation_details"] == {}
    assert result["reservation"] is None
    assert result["route"] is None
    assert result["messages"] == tool_message
    assert all(isinstance(m, RemoveMessage) for m in result["data_recording_messages"])
    assert [m.id for m in result["data_recording_messages"]] == [
        state["data_recording_messages"][0].id,
        state["data_recording_messages"][1].id,
    ]
    assert [m.id for m in result["reservation_messages"]] == [
        state["reservation_messages"][0].id
    ]


def test_submit_reservation_admin_node_handles_mcp_error():
    tool_message = static_tool_message({"is_error": True, "error_message": "boom"})
    state = {
        "data_recording_messages": [tool_message],
        "reservation_messages": [],
    }

    result = submit_reservation_admin_node(state)

    assert result["reservation_id"] is None
    assert result["reservation_phase"] == ReservationPhase.COMPLETED


def test_submit_reservation_admin_node_handles_missing_tool_message():
    state = {
        "data_recording_messages": [HumanMessage("no tool response here")],
        "reservation_messages": [],
    }

    result = submit_reservation_admin_node(state)

    assert result["reservation_id"] is None
    assert result["reservation_phase"] is None
    assert result["messages"] == [
        AIMessage("Failed to create reservation. Please try again later.")
    ]


# after_request_admin_approval_router
def test_after_request_admin_approval_router_approved():
    result = after_request_admin_approval_router({"reservation_status": ReservationStatus.APPROVED})

    assert result == "data_recording_agent"


def test_after_request_admin_approval_router_rejected():
    result = after_request_admin_approval_router({"reservation_status": ReservationStatus.REJECTED})

    assert result == "process_admin_rejection"

from datetime import datetime, timedelta
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage

from chatbot.nodes import (
    after_reservation_router,
    ask_missed_reservation_details_node,
    confirm_reservation_details_node,
    create_reservation_node,
    finalize_reservation_client_node,
)
from chatbot.states import Reservation, ReservationPhase, ReservationUpdates


def static_updates(**overrides) -> ReservationUpdates:
    fields = dict(intent="provide")
    fields.update(overrides)
    return ReservationUpdates(**fields)


def future_ts(hours: int) -> datetime:
    return datetime.now() + timedelta(hours=hours)


def static_reservation_details(**overrides) -> dict:
    fields = dict(
        customer_full_name="John Smith",
        level="B1",
        space_type="EV",
        start_datetime=future_ts(24),
        end_datetime=future_ts(34),
        license_plate="LV-1234",
    )
    fields.update(overrides)
    return fields


def static_reservation(**overrides) -> Reservation:
    return Reservation(**static_reservation_details(**overrides))


# create_reservation_node
def test_create_reservation_node_skips_extraction_without_human_message():
    with patch("chatbot.nodes.__RESERVATION_TURN_EXTRACTOR") as mock_extractor:
        result = create_reservation_node({"messages": [], "reservation_details": {}})

    mock_extractor.invoke.assert_not_called()
    assert result["intent"] == "other"
    assert result["reservation_details"] == {}
    assert "missing customer/user full name" in "; ".join(result["reservation_issues"]) \
        or any("missing" in issue for issue in result["reservation_issues"])



def test_create_reservation_node_merges_extracted_updates():
    updates = static_updates(license_plate="LV-9999", intent="provide")
    state = {
        "messages": [HumanMessage("my plate is LV-9999")],
        "reservation_details": static_reservation_details(),
    }
    with patch("chatbot.nodes.__RESERVATION_TURN_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.return_value = updates
        result = create_reservation_node(state)

    assert result["reservation_details"]["license_plate"] == "LV-9999"
    assert result["reservation_issues"] == []
    assert result["intent"] == "provide"
    assert result["changed_this_turn"] is True


def test_create_reservation_node_reports_no_change_when_values_are_identical():
    details = static_reservation_details()
    state = {
        "messages": [HumanMessage("yes that's right")],
        "reservation_details": details,
    }
    updates = static_updates(**details, intent="confirm")
    with patch("chatbot.nodes.__RESERVATION_TURN_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.return_value = updates
        result = create_reservation_node(state)

    assert result["changed_this_turn"] is False
    assert result["intent"] == "confirm"


def test_create_reservation_node_placeholder_values_do_not_overwrite_existing_details():
    state = {
        "messages": [HumanMessage("uh i don't know the plate")],
        "reservation_details": static_reservation_details(),
    }
    updates = static_updates(license_plate="unknown", intent="provide")
    with patch("chatbot.nodes.__RESERVATION_TURN_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.return_value = updates
        result = create_reservation_node(state)

    assert result["reservation_details"]["license_plate"] == "LV-1234"
    assert result["changed_this_turn"] is False


def test_create_reservation_node_reports_issues_for_bad_dates():
    state = {
        "messages": [HumanMessage("book it end before start")],
        "reservation_details": static_reservation_details(),
    }
    updates = static_updates(
        start_datetime=future_ts(34),
        end_datetime=future_ts(24),
    )
    with patch("chatbot.nodes.__RESERVATION_TURN_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.return_value = updates
        result = create_reservation_node(state)

    assert any("end datetime must be after" in issue for issue in result["reservation_issues"])


# ask_missed_reservation_details_node
def test_ask_missed_reservation_details_node_returns_llm_response():
    response = AIMessage("What's the license plate?")
    state = {
        "reservation_details": static_reservation_details(license_plate=None),
        "reservation_issues": ["missing plate"],
    }
    with patch("chatbot.nodes.__RESERVATION_LLM") as mock_llm:
        mock_llm.invoke.return_value = response
        result = ask_missed_reservation_details_node(state)

    assert result["reservation_phase"] == ReservationPhase.COLLECTING
    assert result["messages"] == [response]
    assert result["reservation_messages"] == [response]


# confirm_reservation_details_node
def test_confirm_reservation_details_node_summarizes_and_asks_to_confirm():
    state = {"reservation_details": static_reservation_details()}

    result = confirm_reservation_details_node(state)

    assert result["reservation_phase"] == ReservationPhase.CONFIRMING
    message = result["messages"][0]
    assert isinstance(message, AIMessage)
    assert "Customer: John Smith" in message.content
    assert "Book it? (yes / cancel)" in message.content
    assert result["reservation_messages"] == [message]


# finalize_reservation_client_node
def test_finalize_reservation_client_node_uses_structured_extractor():
    reservation = static_reservation()
    state = {
        "reservation_messages": [HumanMessage("yes, confirm")],
        "reservation_details": static_reservation_details(),
    }
    with patch("chatbot.nodes.__RESERVATION_DETAILS_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.return_value = reservation
        result = finalize_reservation_client_node(state)

    mock_extractor.invoke.assert_called_once()
    assert result == {
        "reservation": reservation,
        "reservation_phase": ReservationPhase.REGUESTING_APROVAL,
    }


def test_finalize_reservation_client_node_falls_back_to_reservation_details():
    state = {
        "reservation_messages": [HumanMessage("yes, confirm")],
        "reservation_details": static_reservation_details(),
    }
    with patch("chatbot.nodes.__RESERVATION_DETAILS_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.side_effect = Exception("extraction failed")
        result = finalize_reservation_client_node(state)

    assert result["reservation_phase"] == ReservationPhase.REGUESTING_APROVAL
    assert result["reservation"].license_plate == "LV-1234"


def test_finalize_reservation_client_node_recovers_when_both_fallbacks_fail():
    state = {
        "reservation_messages": [HumanMessage("yes, confirm")],
        "reservation_details": {"license_plate": "LV-1234"},  # missing required fields
    }
    with patch("chatbot.nodes.__RESERVATION_DETAILS_EXTRACTOR") as mock_extractor:
        mock_extractor.invoke.side_effect = Exception("extraction failed")
        result = finalize_reservation_client_node(state)

    assert result["reservation_phase"] == ReservationPhase.COLLECTING
    assert "missing Customer" in result["reservation_issues"]
    assert result["messages"] == [AIMessage("Something didn't validate — let's fix it.")]


# after_reservation_router
def test_after_reservation_router_asks_for_missing_details():
    result = after_reservation_router({"reservation_issues": ["missing plate"]})

    assert result == "ask_missed_reservation_details"


def test_after_reservation_router_finalizes_on_plain_confirmation():
    state = {
        "reservation_issues": [],
        "reservation_phase": ReservationPhase.CONFIRMING,
        "changed_this_turn": False,
        "intent": "confirm",
    }

    result = after_reservation_router(state)

    assert result == "finalize_reservation_client"


def test_after_reservation_router_reconfirms_when_details_changed():
    state = {
        "reservation_issues": [],
        "reservation_phase": ReservationPhase.CONFIRMING,
        "changed_this_turn": True,
        "intent": "confirm",
    }

    result = after_reservation_router(state)

    assert result == "confirm_reservation_details"


def test_after_reservation_router_reconfirms_when_not_yet_confirming():
    state = {
        "reservation_issues": [],
        "reservation_phase": ReservationPhase.COLLECTING,
        "changed_this_turn": False,
        "intent": "provide",
    }

    result = after_reservation_router(state)

    assert result == "confirm_reservation_details"

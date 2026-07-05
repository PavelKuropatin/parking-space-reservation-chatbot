from typing import Optional

from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from pydantic import create_model

from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db
from chatbot.prompts import (
    GUARDRAIL_INPUT_BLOCK_MSG,
    GUARDRAIL_OUTPUT_BLOCK_MSG,
    PARSE_RESERVATION_DETAILS_PROMT_TMPL,
    RAG_SUMMARIZATION_PROMPT_TMPL,
    RETRIEVE_RESERVATION_DETAILS_PROMT_TMPL,
    URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL,
)
from chatbot.states import (
    RESERVATION_FIELD_DESCRIPTIONS,
    RESERVATION_FIELD_LABELS,
    ReservationStatus,
    UserConfirmDecision,
    GraphState,
    UserIntentDecision,
    ReservationPhase,
)
from chatbot.utils.graph_utils import get_llm, last_ai, now
from chatbot.guardrail.filtering import get_guardrail
from chatbot.logging import logger

# --------------------------------------------------------------------------- #
# LLM-s
# --------------------------------------------------------------------------- #
__llm = get_llm()
__user_input_classifier = __llm.with_structured_output(UserIntentDecision)
__confirmation_classifier = __llm.with_structured_output(UserConfirmDecision)


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #
def input_guardrail_node(state: GraphState) -> dict:
    human_message = state["human_message"]
    if not human_message:
        return {"input_blocked": False, "block_reason": ""}

    guardrail = get_guardrail()
    result = guardrail.for_input(human_message)
    if result.blocked:
        return {"input_blocked": True, "block_reason": result.entity_summary}

    return {"input_blocked": False, "block_reason": ""}


def blocked_response_node(state: GraphState) -> dict:
    block_reason = state["block_reason"]
    guardrail_message = GUARDRAIL_INPUT_BLOCK_MSG.format(details=block_reason)
    return {"ai_message": guardrail_message}


def output_guardrail_node(state: GraphState) -> dict:
    human_message = state.get("human_message", None)
    ai_message = state.get("ai_message", None)
    if not ai_message:
        # is it possible?
        return {}

    guardrail = get_guardrail()
    result = guardrail.for_input(ai_message)
    if result.blocked:
        guardrail_message = GUARDRAIL_OUTPUT_BLOCK_MSG.format(
            details=result.entity_summary
        )
        # skip human message for history
        return {
            "messages": [AIMessage(guardrail_message)],
            "human_message": None,
            "ai_message": None,
        }

    # everything is ok
    return {
        "messages": [
            HumanMessage(human_message),
            AIMessage(ai_message),
        ],
        "human_message": None,
        "ai_message": None,
    }


# --------------------------------------------------------------------------- #
# ROOT CLASSIFICATION
# --------------------------------------------------------------------------- #
def classify_user_intent_node(state: GraphState) -> dict:
    question = state["human_message"]

    messages = URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL.invoke({"question": question})
    decision = __user_input_classifier.invoke(messages)

    return {"route": decision.route}


def unknown_node(state: GraphState) -> dict:
    if state.get("ai_message", None):
        return {}

    greetings = (
        "Hello! I'm here to help you reserve a parking spot quickly and easily.\n"
        "Just tell me what you need — I'll collect the details, then our team will confirm your booking shortly."
    )
    return {"ai_message": greetings}


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def qa_system_rag_node(state: GraphState) -> dict:
    question = state["human_message"]

    try:
        db = get_parking_data_db()
        objects = get_parking_info_retriever().query(question).objects
        documents = [obj.properties["content"] for obj in objects]
        messages = RAG_SUMMARIZATION_PROMPT_TMPL.invoke(
            {
                "rag_context": "\n'n".join(documents),
                "pricing": db.get_space_pricing(),
                "working_hours": db.get_working_hours(),
                "question": question,
            }
        )
        response = __llm.invoke(messages)
        return {"ai_message": response.content}
    except Exception as e:
        logger.error("Failed to execute RAG node: %s", e)
        return {"ai_message": "Failed to request information. Please try again later"}


# --------------------------------------------------------------------------- #
# Reservation
# --------------------------------------------------------------------------- #
def extract_reservation_details(
    human_message: str,
    prev_ai_message: str,
    reservation_details: dict,
    missed_details: dict,
) -> dict:
    if not missed_details:
        return reservation_details

    updated = reservation_details.copy()

    for field, field_description in missed_details.items():
        field_model = create_model(field, **{field: (Optional[str], None)})
        messages = PARSE_RESERVATION_DETAILS_PROMT_TMPL.invoke(
            {
                "field": field,
                "field_description": field_description,
                "ai_message": prev_ai_message,
                "human_message": human_message,
                "now": now(),
            }
        )
        response = __llm.with_structured_output(field_model).invoke(messages)
        field_value = getattr(response, field, None)
        if field_value and field_value != "null":
            updated[field] = field_value

    return updated


def create_reservation_node(state: GraphState) -> dict:
    reservation_phase = state.get("reservation_phase")
    reservation_details = state.get("reservation_details", {})
    human_message = state["human_message"]
    prev_ai = last_ai(state["messages"])
    prev_ai_message = prev_ai.content if prev_ai else ""

    missed_details = {
        f: RESERVATION_FIELD_DESCRIPTIONS[f]
        for f in RESERVATION_FIELD_LABELS
        if not reservation_details.get(f)
    }

    # data collection
    if missed_details or reservation_phase == ReservationPhase.COLLECTING:
        # update details state
        reservation_details = extract_reservation_details(
            human_message, prev_ai_message, reservation_details, missed_details
        )
        missed_details = {
            f: RESERVATION_FIELD_DESCRIPTIONS[f]
            for f in RESERVATION_FIELD_LABELS
            if not reservation_details.get(f)
        }

        if missed_details:
            messages = RETRIEVE_RESERVATION_DETAILS_PROMT_TMPL.invoke(
                {
                    "reservation_details": reservation_details,
                    "gaps": missed_details.keys(),
                    "now": now(),
                    "human_message": human_message,
                }
            )
            response = __llm.invoke(messages)
            return {
                "reservation_phase": ReservationPhase.COLLECTING,
                "reservation_details": reservation_details,
                "ai_message": response.content,
            }

        # all data gathered
        summary = summarize_reservation(reservation_details)
        message = "Please confirm:\n" f"{summary}\n" "Book it? (yes / cancel)"
        return {
            "reservation_phase": ReservationPhase.CONFIRMING,
            "reservation_details": reservation_details,
            "ai_message": message,
        }

    # expecet confirmation message
    response = __confirmation_classifier.invoke(
        [
            SystemMessage("""
            User was asked to confirm a parking booking. Classify reply.
            - yes: The user explicitly confirms, agrees, approves, or wants to proceed.
            - cancel: The user explicitly rejects, declines, stops, aborts, or does not want to proceed.
            - unclear: The user message is a unclear
            """),
            HumanMessage(human_message),
        ]
    )
    match response.decision:
        case "yes":
            return {
                "reservation_phase": ReservationPhase.REGUESTING_APROVAL,
            }
        case "cancel":
            cancellation_message = "Cancelled — nothing was booked. Anything else?"
            return {
                "route": None,
                "reservation_details": {},
                "reservation_phase": ReservationPhase.CANCELLED,
                "ai_message": cancellation_message,
            }
        case "unclear":
            return {
                "ai_message": "Please correct your answer",
            }


def summarize_reservation(reservation_details: dict) -> str:
    return "\n".join(
        f" - {label}: {reservation_details[field]}"
        for field, label in RESERVATION_FIELD_LABELS.items()
        if reservation_details.get(field) is not None
    )


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
def request_admin_approval_node(state: GraphState, config: RunnableConfig) -> dict:
    calling_thread_id = config["configurable"]["thread_id"]
    reservation_details = state["reservation_details"]

    try:
        request_id = f"admin-request-{now("%Y%m%dT%H%M%S")}"
        interrupt_payload = {
            "calling_thread_id": calling_thread_id,
            "request_id": request_id,
            "message": ("Details:\n" f"{summarize_reservation(reservation_details)}"),
        }
        response = interrupt(interrupt_payload)
        reservation_status = ReservationStatus(response["reservation_status"].lower())
        reservation_details |= {"approved_ts": response["approval_ts"]}
        return {
            "reservation_status": reservation_status,
            "reservation_details": reservation_details,
        }
    except GraphInterrupt as e:
        raise e
    except Exception as e:
        logger.error("Failed to request admin approval: %s", e)
        return {
            "route": None,
            "reservation_phase": None,
            "reservation_details": {},
            "ai_message": "Failed to request admin approval. Please try again later",
        }


def process_admin_rejection_node(_state: GraphState) -> dict:
    return {
        "route": None,
        "reservation_phase": ReservationPhase.COMPLETED,
        "reservation_details": {},
        "ai_message": "Your reservaton was rejected",
    }


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
def after_input_guardrail_router(state: GraphState) -> str:
    if state.get("input_blocked", False):
        return "blocked_response"

    route = state.get("route", "classify_intent")
    if route == "reservation":
        return "reservation"
    return "classify_intent"


def after_classify_intent_router(state: GraphState) -> str:
    route = state.get("route")
    if route == "information":
        return "information"
    if route == "reservation":
        return "reservation"
    return "_unknown"


def after_reservation_router(state: GraphState) -> str:
    reservation_phase = state.get("reservation_phase", None)
    if reservation_phase == ReservationPhase.REGUESTING_APROVAL:
        return "request_admin_approval"
    return "output_guardrail"


def after_request_admin_approval_router(state: GraphState) -> str:
    reservation_status = state["reservation_status"]
    match reservation_status:
        case ReservationStatus.APPROVED:
            return "process_admin_approval"
        case ReservationStatus.REJECTED:
            return "process_admin_rejection"

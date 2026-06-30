from typing import Optional

from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import create_model

from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db
from chatbot.graph.prompts import (
    GUARDRAIL_INPUT_BLOCK_MSG,
    GUARDRAIL_OUTPUT_BLOCK_MSG,
    PARSE_RESERVATION_DETAILS_PROMT_TMPL,
    RAG_SUMMARIZATION_PROMPT_TMPL,
    RETRIEVE_RESERVATION_DETAILS_PROMT_TMPL,
    URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL,
)
from chatbot.graph.states import (
    RESERVATION_FIELD_DESCRIPTIONS,
    RESERVATION_FIELD_LABELS,
    UserConfirmDecision,
    GraphState,
    UserIntentDecision,
)
from chatbot.graph.utils import get_llm, now
from chatbot.guardrail.filtering import get_guardrail

# --------------------------------------------------------------------------- #
# LLM-s
# --------------------------------------------------------------------------- #
__llm = get_llm()
__user_input_classifier = __llm.with_structured_output(UserIntentDecision)
__user_reservation_llm = __llm
__confirmation_classifier = __llm.with_structured_output(UserConfirmDecision)


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #
def input_guardrail_node(state: GraphState) -> dict:
    human_message = state.get("human_message", "")
    if not human_message:
        return {"input_blocked": False, "block_reason": ""}

    guardrail = get_guardrail()
    result = guardrail.for_input(human_message)
    if result.blocked:
        return {"input_blocked": True, "block_reason": result.entity_summary}

    return {"input_blocked": False, "block_reason": ""}


def blocked_response_node(state: GraphState) -> dict:
    block_reason = state.get("block_reason", "Message could not be processed.")
    guardrail_message = GUARDRAIL_INPUT_BLOCK_MSG.format(details=block_reason)
    return {"ai_message": guardrail_message}


def output_guardrail_node(state: GraphState) -> dict:

    human_message = state.get("human_message", "")
    ai_message = state.get("ai_message", "")
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
        return {"messages": [AIMessage(guardrail_message)]}

    # everything is ok
    return {
        "messages": [
            HumanMessage(human_message),
            AIMessage(ai_message),
        ]
    }


# --------------------------------------------------------------------------- #
# ROOT CLASSIFICATION
# --------------------------------------------------------------------------- #
def __refresh_reservation_status() -> dict:
    return {
        "current_details": {},
        "reservation_phase": "collecting",
    }


def classify_user_intent_node(state: GraphState) -> dict:
    question = state.get("human_message", "")

    messages = URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL.invoke({"question": question})
    decision = __user_input_classifier.invoke(messages)

    updates = {"route": decision.route}
    # reset previous state
    if decision.route == "reservation" and state.get("reservation_phase") in (
        None,
        "done",
        "cancelled",
    ):
        updates |= __refresh_reservation_status()
    return updates


def unknown_node(_state: GraphState) -> dict:
    greetings = (
        "Hello! I'm here to help you reserve a parking spot quickly and easily.\n"
        "Just tell me what you need — I'll collect the details, then our team will confirm your booking shortly."
    )
    return {"ai_message": greetings}


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def qa_system_rag_input_node(state: GraphState) -> dict:
    question = state.get("human_message", "")
    documents = get_parking_info_retriever().query(question).objects
    rag_context = [document.properties["content"] for document in documents]
    return {"rag_context": rag_context}


def qa_system_rag_output_node(state: GraphState) -> dict:
    question = state.get("human_message", "")
    rag_context = state.get("rag_context", [])

    # TODO try catch
    db = get_parking_data_db()
    messages = RAG_SUMMARIZATION_PROMPT_TMPL.invoke(
        {
            "rag_context": rag_context,
            "pricing": db.get_space_pricing(),
            "working_hours": db.get_working_hours(),
            "available_spaces": db.get_avaliable_spaces(),
            "question": question,
        }
    )
    response = __llm.invoke(messages)
    return {"ai_message": response.content}


# --------------------------------------------------------------------------- #
# Reservation
# --------------------------------------------------------------------------- #


def extract_reservation_details(
    human_message: str, prev_ai_message: str, current_details: dict, missed_details: dict
) -> dict:
    if not missed_details:
        return current_details

    updated = current_details.copy()

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
        response = __user_reservation_llm.with_structured_output(field_model).invoke(
            messages
        )
        field_value = getattr(response, field, None)
        if field_value:
            updated[field] = field_value

    return updated


def extract_reservation_details_node(state: GraphState) -> dict:
    human_message = state.get("human_message", "")
    prev_ai_message = state.get("ai_message", "")
    current_details = state.get("current_details", {})
    missed_details = {
        f: RESERVATION_FIELD_DESCRIPTIONS[f]
        for f in RESERVATION_FIELD_LABELS
        if not current_details.get(f)
    }
    reservetion_phase = state.get("reservation_phase", "")

    # data collection
    if missed_details or reservetion_phase == "collecting":
        # update details state
        current_details = extract_reservation_details(
            human_message, prev_ai_message, current_details, missed_details
        )
        missed_details = {
            f: RESERVATION_FIELD_DESCRIPTIONS[f]
            for f in RESERVATION_FIELD_LABELS
            if not current_details.get(f)
        }

        if missed_details:
            messages = RETRIEVE_RESERVATION_DETAILS_PROMT_TMPL.invoke(
                {
                    "current_details": current_details,
                    "gaps": missed_details.keys(),
                    "now": now(),
                    "human_message": human_message,
                }
            )
            response = __user_reservation_llm.invoke(messages)
            return {
                **state,
                "current_details": current_details,
                "reservation_phase": "collecting",
                "ai_message": response.content,
            }

        # all data gathered
        summary = summarize_reservation(current_details)
        message = f"Please confirm:\n{summary}\n\nBook it? (yes / cancel / change)"
        return {
            **state,
            "current_details": current_details,
            "reservation_phase": "confirming",
            "ai_message": message,
        }

    # expecet confirmation message confirmation
    response = __confirmation_classifier.invoke(
        [
            SystemMessage("""
            User was asked to confirm a parking booking. Classify reply.
            - yes: The user confirms, agrees, approves, or wants to proceed.
            - cancel: The user rejects, declines, stops, aborts, or does not want to proceed.
            - change: The user wants to modify, edit, adjust, change something before proceeding or unclear response
            """),
            HumanMessage(human_message),
        ]
    )
    match response.decision:
        case "yes":
            reservation_id = save_reservation(state["current_details"])
            confirmed_reservation_message = f"Booked. Reference: {reservation_id}"
            return {
                **state,
                "ai_message": confirmed_reservation_message,
                "reservation_phase": "done",
                "route": None,
                "reservation_id": reservation_id
            }
        case "cancel":
            cancellation_message = "Cancelled — nothing was booked. Anything else?"
            return {
                **state,
                "ai_message": cancellation_message,
                "reservation_phase": "cancelled",
                "route": None,
                "current_details": {},
            }
        case "change":
            messages = RETRIEVE_RESERVATION_DETAILS_PROMT_TMPL.invoke(
                {
                    "reservation_phase": "collecting",
                    "current_details": current_details,
                    "gaps": missed_details,
                    "now": now(),
                    "human_message": human_message,
                }
            )
            response = __user_reservation_llm.invoke(messages)
            return {
                **state,
                "current_details": current_details,
                "response": response,
                "reservation_phase": "collection",
            }


def summarize_reservation(current_details: dict) -> str:
    return "\n".join(
        f" - {label}: {current_details[field]}"
        for field, label in RESERVATION_FIELD_LABELS.items()
        if current_details.get(field) is not None
    )


def save_reservation(current_details: dict) -> int:
    response = get_parking_data_db().write_reservation(current_details)
    reservation_id = response[0]["id"]
    return reservation_id


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
    if route == "information_request":
        return "information_request"
    if route == "reservation":
        return "reservation"
    return "unknown"

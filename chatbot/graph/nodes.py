from datetime import datetime

from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db
from chatbot.graph.models import (
    RESERVATION_DETAILS_SPECS,
    RESERVATION_DETAILS_SPEC_BY_NAME,
)
from chatbot.graph.prompts import (
    GUARDRAIL_INPUT_BLOCK_MSG,
    GUARDRAIL_OUTPUT_BLOCK_MSG,
    RAG_SUMMARIZATION_PROMPT_TMPL,
    RESERVATION_DETAILS_PARSING_PROMT_TMPL,
    URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL,
)
from chatbot.graph.states import (
    RESERVATION_FIELD_LABELS,
    UserConfirmDecision,
    GraphState,
    UserIntentDecision,
    ParkingReservationDetails,
)
from chatbot.graph.utils import get_llm, last_ai_output, last_user_input
from chatbot.guardrail.filtering import get_guardrail

# --------------------------------------------------------------------------- #
# LLM-s
# --------------------------------------------------------------------------- #
__llm = get_llm()
__user_input_classifier = __llm.with_structured_output(UserIntentDecision)
__user_info_extactor = __llm.with_structured_output(ParkingReservationDetails)
__confirmation_classifier = __llm.with_structured_output(UserConfirmDecision)


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #
def input_guardrail_node(state: GraphState) -> dict:

    # possible pii already in history
    messages = state.get("messages", [])
    last_user = last_user_input(messages)

    if not last_user:
        return {"input_blocked": False, "block_reason": ""}

    guardrail = get_guardrail()
    result = guardrail.for_input(last_user)
    if result.blocked:
        return {"input_blocked": True, "block_reason": result.entity_summary}

    return {"input_blocked": False, "block_reason": ""}


def blocked_response_node(state: GraphState) -> dict:
    reason = state.get("block_reason", "Message could not be processed.")

    message = GUARDRAIL_INPUT_BLOCK_MSG.format(details=reason)
    return {"messages": [AIMessage(content=message)]}


def output_guardrail_node(state: GraphState) -> dict:
    messages = state.get("messages", [])
    # possible pii already in history
    last_ai = last_ai_output(messages)
    if not last_ai:
        return {}

    guardrail = get_guardrail()
    result = guardrail.for_input(last_ai)
    if result.blocked:
        message = GUARDRAIL_OUTPUT_BLOCK_MSG.format(details=result.entity_summary)
        return {"messages": [AIMessage(message)]}

    return {}


# --------------------------------------------------------------------------- #
# ROOT CLASSIFICATION
# --------------------------------------------------------------------------- #
def __refresh_reservation_status() -> dict:
    return {
        "current_details": {},
        "errors": {},
        "pending_field": None,
        "pending_error": None,
        "intent": None,
        "reservation_phase": "collecting",
    }


def classify_user_intent_node(state: GraphState) -> dict:
    question = last_user_input(state["messages"])
    messages = URER_INPUT_ROOT_CLASSIFICATION_PROMPT_TMPL.invoke({"text": question})
    decision = __user_input_classifier.invoke(messages)
    updates = {"route": decision.route}
    if decision.route == "reservation" and state.get("reservation_phase") in (
        None,
        "done",
        "cancelled",
    ):
        updates |= __refresh_reservation_status()
    return updates


def unknown_node(_state: GraphState) -> dict:
    message = (
"Hello! I'm here to help you reserve a parking spot quickly and easily.\n"
"Just tell me what you need — I'll collect the details, then our team will confirm your booking shortly."
    )
    return {"messages": AIMessage(message)}


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def qa_system_rag_input_node(state: GraphState) -> dict:
    question = last_user_input(state["messages"])
    documents = get_parking_info_retriever().query(question).objects
    rag_context = [document.properties["content"] for document in documents]
    return {"rag_context": rag_context}


def qa_system_rag_output_node(state: GraphState) -> dict:
    question = last_user_input(state["messages"])
    rag_context = state.get("rag_context", [])

    # TODO try catch
    db = get_parking_data_db()
    messages = RAG_SUMMARIZATION_PROMPT_TMPL.invoke(
        {
            "rag_context": rag_context, 
            "pricing": db.get_space_pricing(), 
            "working_hours": db.get_working_hours(), 
            "available_spaces": db.get_avaliable_spaces(), 
            "question": question
        }
    )
    response = __llm.invoke(messages)
    return {"messages": [AIMessage(response.content)]}


# --------------------------------------------------------------------------- #
# Reservation
# --------------------------------------------------------------------------- #
def extract_reservation_details_node(state: GraphState) -> dict:
    current_details = state.get("current_details", {})

    messages = RESERVATION_DETAILS_PARSING_PROMT_TMPL.invoke(
        {
            "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "current_details": current_details,
            "history": state["messages"][-2:],
        }
    )
    extracted_details = __user_info_extactor.invoke(messages).model_dump()
    merged = dict(current_details)
    for field, value in extracted_details.items():
        if value is not None and value != merged.get(field):
            merged[field] = value
    return {"current_details": merged, "reservation_phase": "collecting"}


def validate_reservation_details(state: GraphState) -> dict:
    current_details = state["current_details"]
    errors: dict[str, str] = {}
    for spec in RESERVATION_DETAILS_SPECS:
        val = current_details.get(spec.name)
        if val is not None:
            err = spec.validate(val, current_details)
            if err:
                errors[spec.name] = err

    pending, pending_error = None, None
    for spec in RESERVATION_DETAILS_SPECS:
        if spec.name in errors:
            pending, pending_error = spec.name, errors[spec.name]
            break
        if spec.required and not current_details.get(spec.name):
            pending, pending_error = spec.name, None
            break
    return {"errors": errors, "pending_field": pending, "pending_error": pending_error}


def ask_missed_reservation_details_node(state: GraphState) -> dict:
    spec = RESERVATION_DETAILS_SPEC_BY_NAME[state["pending_field"]]
    text = (
        f"{state['pending_error']} {spec.prompt}"
        if state.get("pending_error")
        else spec.prompt
    )
    return {"messages": [AIMessage(text)]}


def summarize_booking(current_details: dict) -> str:
    return "\n".join(
        f" - {RESERVATION_FIELD_LABELS[k]}: {current_details[k]}"
        for k in RESERVATION_FIELD_LABELS # pylint: disable=consider-using-dict-items
        if current_details.get(k) is not None
    )


def request_reservation_confirmation_node(state: GraphState) -> dict:
    body = summarize_booking(state["current_details"])
    return {
        "messages": [
            AIMessage(f"Please confirm:\n{body}\n\nBook it? (yes / cancel / change)")
        ],
        "reservation_phase": "confirming",
    }


def interpret_user_confirmation_node(state: GraphState) -> dict:
    user_text = last_user_input(state["messages"])
    decision = __confirmation_classifier.invoke(
        [
            SystemMessage(
                "User was asked to confirm a parking booking. Classify reply."
            ),
            HumanMessage(user_text),
        ]
    )
    return {"intent": decision.intent}


def cancel_inprogress_reservation_node(_state: GraphState) -> dict:
    return {
        "messages": [AIMessage("Cancelled — nothing was booked. Anything else?")],
        "reservation_phase": "cancelled",
        "current_details": {},
        "errors": {},
        "pending_field": None,
        "pending_error": None,
        "intent": None,
    }


def finalize_reservation_node(state: GraphState) -> dict:
    reference_id = book_reservation(state["current_details"])
    return {
        "messages": [AIMessage(f"Booked. Reference: {reference_id}")],
        "reservation_phase": "done",
    }


def book_reservation(_current_details: dict) -> str:
    return "PRK-" + datetime.now().strftime("%y%m%d%H%M%S")


# --------------------------------------------------------------------------- #
# Routers
# --------------------------------------------------------------------------- #
def after_guardrail_router(state: GraphState) -> str:
    if state.get("input_blocked", False):
        return "blocked_response"

    phase = state.get("reservation_phase")
    if phase == "confirming":
        return "interpret_user_confirmation"
    if phase == "collecting":
        return "extract_details"
    return "classify_intent"


def intent_router(state: GraphState) -> str:
    route = state.get("route")
    if route == "information_request":
        return "information_request"
    if route == "reservation":
        return "extract_details"
    return "unknown"


def missed_reservation_details_router(state: GraphState) -> str:
    return (
        "ask_missed_details"
        if state.get("pending_field")
        else "request_user_confirmation"
    )


def user_confirmation_router(state: GraphState) -> str:
    return {
        "yes": "finalize_reservation",
        "change": "extract_details",
        "cancel": "cancel_reservation",
    }.get(state.get("intent"), "request_user_confirmation")

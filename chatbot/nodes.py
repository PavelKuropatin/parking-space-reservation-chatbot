from datetime import datetime
import json

from langchain.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db
from chatbot.prompts import (
    REQUEST_USER_DATA_PROMPT_TMPL,
    EXTRACT_ITERATION_UPDATES_PROMPT_TMPL,
    EXTRACT_LAST_STATE_PROMPT_TMPL,
    GUARDRAIL_INPUT_BLOCK_MSG,
    GUARDRAIL_OUTPUT_BLOCK_MSG,
    RAG_SUMMARIZATION_PROMPT_TMPL,
    USER_INPUT_CLASSSIFICATION_PROMPT_TMPL,
)
from chatbot.states import (
    RESERVATION_FIELD_LABELS,
    Reservation,
    ReservationStatus,
    ReservationUpdates,
    GraphState,
    UserInputType,
    ReservationPhase,
)
from chatbot.utils.graph_utils import get_llm, last_ai, last_human, now
from chatbot.guardrail.filtering import get_guardrail
from chatbot.logging import logger

# --------------------------------------------------------------------------- #
# LLM-s
# --------------------------------------------------------------------------- #
__zero_temp_llm = get_llm(temperature=0)
__RAG_LLM = __zero_temp_llm
__USER_INPUT_CLASSIFIER = __zero_temp_llm.with_structured_output(UserInputType)

__default = get_llm(temperature=0.3)
__RESERVATION_LLM = __default
__RESERVATION_TURN_EXTRACTOR = __default.with_structured_output(ReservationUpdates)
__RESERVATION_DETAILS_EXTRACTOR = __default.with_structured_output(Reservation)


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

    return {
        "input_blocked": False,
        "block_reason": "",
        "messages": [HumanMessage(human_message)],
    }


def blocked_response_node(state: GraphState) -> dict:
    block_reason = state["block_reason"]
    guardrail_message = GUARDRAIL_INPUT_BLOCK_MSG.format(details=block_reason)
    return {"messages": [AIMessage(guardrail_message)]}


def output_guardrail_node(state: GraphState) -> dict:
    ai_message = last_ai(state.get("messages", []))
    if not ai_message:
        return {}

    guardrail = get_guardrail()
    result = guardrail.for_output(ai_message.content)
    if result.blocked:
        guardrail_message = GUARDRAIL_OUTPUT_BLOCK_MSG.format(
            details=result.entity_summary
        )
        return {
            "messages": [RemoveMessage(id=ai_message.id), AIMessage(guardrail_message)],
            "human_message": None,
        }

    # everything is ok
    return {
        "human_message": None,
    }


# --------------------------------------------------------------------------- #
# ROOT CLASSIFICATION
# --------------------------------------------------------------------------- #
def classify_user_input_node(state: GraphState) -> dict:
    question = last_human(state.get("messages", []))
    if question is None:
        return {"route": "_unknown"}

    messages = USER_INPUT_CLASSSIFICATION_PROMPT_TMPL.invoke({"question": question})
    response: UserInputType = __USER_INPUT_CLASSIFIER.invoke(messages)
    match response.route:
        case "information":
            return {
                "route": "information",
                "qa_messages": [question],
            }
        case "reservation":
            return {
                "route": "reservation",
                "reservation_messages": [question],
            }
        case "_unknown":
            return {"route": "_unknown"}


def unknown_node(_state: GraphState) -> dict:

    # if state.get("ai_message", None):
    #     return {}

    greetings = (
        "Hello! I'm here to help you reserve a parking spot quickly and easily.\n"
        "Just tell me what you need — I'll collect the details, then our team will confirm your booking shortly."
    )
    return {"messages": [AIMessage(greetings)]}


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def qa_system_rag_node(state: GraphState) -> dict:
    qa_messages = state.get("qa_messages", [])
    question = last_human(qa_messages)

    try:
        db = get_parking_data_db()
        objects = get_parking_info_retriever().query(question.content).objects
        documents = [obj.properties["content"] for obj in objects]
        messages = RAG_SUMMARIZATION_PROMPT_TMPL.invoke(
            {
                "rag_context": "\n\n".join(documents),
                "pricing": db.get_space_pricing(),
                "working_hours": db.get_working_hours(),
                "qa_conversation": qa_messages,
            }
        )
        response = __RAG_LLM.invoke(messages)
        return {
            "messages": [AIMessage(response.content)],
            "qa_messages": [AIMessage(response.content)],
        }
    except Exception as e:
        logger.error("Failed to execute RAG node: %s", e, exc_info=True)
        return {
            "messages": [
                AIMessage("Failed to request information. Please try again later")
            ]
        }


# --------------------------------------------------------------------------- #
# Reservation
# --------------------------------------------------------------------------- #
def get_reservation_issues(
    reservation_details: dict, current_ts: datetime
) -> list[str]:
    issues = []
    for field, label in RESERVATION_FIELD_LABELS.items():
        if not reservation_details.get(field):
            issues.append(f"missing {label}")

    start_datetime, end_datetime = reservation_details.get(
        "start_datetime"
    ), reservation_details.get("end_datetime")
    if start_datetime and end_datetime and end_datetime <= start_datetime:
        issues.append("the end datetime must be after the start datetime")
    if start_datetime and start_datetime < current_ts:
        issues.append("the start datetime is in the past")
    return issues


def merge_updates(reservation_details: dict, udpates: ReservationUpdates) -> dict:
    new_details = dict(reservation_details)
    for field in RESERVATION_FIELD_LABELS:
        field_value = getattr(udpates, field)
        if field_value is not None:
            new_details[field] = field_value
    return new_details


def summarize_reservation(reservation_details: dict) -> str:
    return "\n".join(
        f" - {label}: {reservation_details[field]}"
        for field, label in RESERVATION_FIELD_LABELS.items()
        if reservation_details.get(field) is not None
    )


__PLACEHOLDERS = {
    "",
    "unknown",
    "none",
    "-",
    "??",
    "null",
}


def clean_updates(updates: ReservationUpdates) -> ReservationUpdates:
    for field in ("customer_full_name", "space_type", "level", "license_plate"):
        value = getattr(updates, field)
        if isinstance(value, str) and value.strip().lower() in __PLACEHOLDERS:
            setattr(updates, field, None)
    for field in ("start_datetime", "end_datetime"):
        value = getattr(updates, field)
        if isinstance(value, datetime) and value.tzinfo is not None:
            setattr(updates, field, value.replace(tzinfo=None))
    return updates


def create_reservation_node(state: GraphState) -> dict:
    current_ts = datetime.now()
    reservation_details = state.get("reservation_details", {})

    # No human turn yet -> nothing to extract.
    if not any(isinstance(m, HumanMessage) for m in state.get("messages", [])):
        return {
            "reservation_details": reservation_details,
            "reservation_issues": get_reservation_issues(
                reservation_details, current_ts
            ),
            "intent": "other",
        }

    messages = EXTRACT_ITERATION_UPDATES_PROMPT_TMPL.invoke(
        {"now": current_ts, "conversation": state["messages"]}
    )
    updates: ReservationUpdates = clean_updates(__RESERVATION_TURN_EXTRACTOR.invoke(messages))
    reservation_details = state.get("reservation_details", {})
    new_reservation_details = merge_updates(reservation_details, updates)
    changed = any(
        reservation_details.get(f) != new_reservation_details.get(f)
        for f in RESERVATION_FIELD_LABELS
    )
    return {
        "reservation_details": new_reservation_details,
        "reservation_issues": get_reservation_issues(
            new_reservation_details, datetime.now()
        ),
        "intent": updates.intent,
        "changed_this_turn": changed,
    }


def _fmt(dt) -> str:
    if not dt:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%a %d %b %Y, %H:%M")


def _summarize_known(reservation_details: dict) -> str:
    parts = []
    if reservation_details.get("customer_full_name"):
        parts.append(f"customer_full_name={reservation_details['customer_full_name']}")
    if reservation_details.get("level"):
        parts.append(f"level={reservation_details['level']}")
    if reservation_details.get("space_type"):
        parts.append(f"space_type={reservation_details['space_type']}")
    if reservation_details.get("start_datetime"):
        parts.append(f"start_datetime={_fmt(reservation_details['start_datetime'])}")
    if reservation_details.get("end_datetime"):
        parts.append(f"end_datetime={_fmt(reservation_details['end_datetime'])}")
    if reservation_details.get("license_plate"):
        parts.append(f"license_plate={reservation_details['license_plate']}")
    return ", ".join(parts) or "nothing yet"


def ask_missed_reservation_details_node(state: GraphState) -> dict:
    reservation_details = state.get("reservation_details", {})
    reservation_issues = state.get("reservation_issues", [])
    messages = REQUEST_USER_DATA_PROMPT_TMPL.invoke(
        {
            "reservation_details": _summarize_known(reservation_details),
            "issues": "; ".join(reservation_issues) or "none",
        }
    )

    response = __RESERVATION_LLM.invoke(messages)
    return {
        "reservation_phase": ReservationPhase.COLLECTING,
        "messages": [response],
        "reservation_messages": [response],
    }


def confirm_reservation_details_node(state: GraphState) -> dict:
    reservation_details = state.get("reservation_details", {})
    summary = summarize_reservation(reservation_details)
    message = "Please confirm:\n" f"{summary}\n" "Book it? (yes / cancel)"
    return {
        "reservation_phase": ReservationPhase.CONFIRMING,
        "messages": [AIMessage(message)],
        "reservation_messages": [AIMessage(message)],
    }


def finalize_reservation_client_node(state: GraphState) -> dict:
    current_ts = datetime.now()
    try:
        messages = EXTRACT_LAST_STATE_PROMPT_TMPL.invoke(
            {"now": now(), "conversation": state.get("reservation_messages", [])}
        )
        reservation = __RESERVATION_DETAILS_EXTRACTOR.invoke(messages)
    except Exception:
        # Fall back to the values we already collected & validated.
        try:
            reservation = Reservation(
                **{
                    k: state["reservation_details"].get(k)
                    for k in RESERVATION_FIELD_LABELS
                }
            )
        except Exception:
            reservation_issues = get_reservation_issues(state["reservation_details"], current_ts) or [
                "please re-enter the dates"
            ]
            return {
                "messages": [AIMessage("Something didn't validate — let's fix it.")],
                "reservation_phase": ReservationPhase.COLLECTING,
                "reservation_issues": reservation_issues,
            }

    return {
        "reservation": reservation,
        "reservation_phase": ReservationPhase.REGUESTING_APROVAL,
    }


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
def request_admin_approval_node(state: GraphState, config: RunnableConfig) -> dict:
    calling_thread_id = config["configurable"]["thread_id"]
    reservation: Reservation = state["reservation"]
    reservation_details = reservation.model_dump(mode="json")

    try:
        request_id = f"admin-request-{now("%Y%m%dT%H%M%S")}"
        interrupt_payload = {
            "calling_thread_id": calling_thread_id,
            "request_id": request_id,
            "message": (
                "Details:\n"
                f"{summarize_reservation(reservation_details)}\n"
                f"{reservation}"
            ),
        }
        response = interrupt(interrupt_payload)
        reservation_status = ReservationStatus(response["reservation_status"].lower())
        reservation_details |= {"approved_ts": response["approval_ts"]}
        return {
            "reservation_status": reservation_status,
            "reservation_details": reservation_details,
            "data_recording_messages": HumanMessage(
                json.dumps(reservation_details, ensure_ascii=False, indent=2)
            ),
        }
    except GraphInterrupt as e:
        raise e
    except Exception as e:
        logger.error("Failed to request admin approval: %s", e, exc_info=True)
        return {
            "route": None,
            "reservation_phase": None,
            "reservation_details": {},
            "reservation": None,
            "messages": [
                AIMessage("Failed to request admin approval. Please try again later")
            ],
        }


def process_admin_rejection_node(_state: GraphState) -> dict:
    return {
        "route": None,
        "reservation_phase": ReservationPhase.COMPLETED,
        "reservation_details": {},
        "reservation": None,
        "messages": [AIMessage("Your reservaton was rejected")],
    }


def submit_reservation_admin_node(state: GraphState) -> dict:
    data_recording_messages = [
        RemoveMessage(id=msg.id) for msg in state["data_recording_messages"]
    ]
    reservation_messages = [
        RemoveMessage(id=msg.id) for msg in state["reservation_messages"]
    ]

    try:
        # process mcp tool response
        mcp_message = next(
            m
            for m in state["data_recording_messages"][::-1]
            if isinstance(m, ToolMessage)
        )
        mcp_response = json.loads(mcp_message.content[0]["text"])
        if mcp_response["is_error"]:
            logger.error(
                "Failed to submit a reservation [%s]", mcp_response["error_message"]
            )
            reserveation_id = None
        else:
            reserveation_id = mcp_response["reservation_id"]

        # output
        agent_response = state["data_recording_messages"][-1]
        return {
            "reservation_messages": reservation_messages,
            "data_recording_messages": data_recording_messages,
            "route": None,
            "reservation_phase": ReservationPhase.COMPLETED,
            "reservation_id": reserveation_id,
            "reservation_details": {},
            "reservation": None,
            "messages": agent_response,
        }
    except Exception as e:
        logger.error("Failed to submit reservation: %s", e, exc_info=True)
        return {
            "reservation_messages": reservation_messages,
            "data_recording_messages": data_recording_messages,
            "route": None,
            "reservation_id": None,
            "reservation_phase": None,
            "reservation_details": {},
            "reservation": None,
            "messages": [
                AIMessage("Failed to create reservation. Please try again later.")
            ],
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
    return "classify_user_input"


def after_classify_user_input_router(state: GraphState) -> str:
    return state.get("route", "_unknown")


def after_reservation_router(state: GraphState) -> dict:
    if state["reservation_issues"]:
        return "ask_missed_reservation_details"

    is_confirming = state.get("reservation_phase") == ReservationPhase.CONFIRMING
    changed = state.get("changed_this_turn", False)
    intent = state.get("intent", "provide")

    # only confirmation message recieved
    if is_confirming and not changed and intent == "confirm":
        return "finalize_reservation_client"

    # try to confirm again
    return "confirm_reservation_details"


def after_request_admin_approval_router(state: GraphState) -> str:
    reservation_status = state["reservation_status"]
    match reservation_status:
        case ReservationStatus.APPROVED:
            return "data_recording_agent"
        case ReservationStatus.REJECTED:
            return "process_admin_rejection"


def after_data_recording_router(state: GraphState) -> str:
    last = state["data_recording_messages"][-1]
    return (
        "data_recording_tools"
        if getattr(last, "tool_calls", None)
        else "_submit_reservation_admin"
    )

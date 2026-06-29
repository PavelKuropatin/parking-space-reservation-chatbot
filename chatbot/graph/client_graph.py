from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from chatbot.graph.nodes import (
    cancel_inprogress_reservation_node,
    after_guardrail_router,
    blocked_response_node,
    ask_missed_reservation_details_node,
    classify_user_intent_node,
    input_guardrail_node,
    output_guardrail_node,
    unknown_node,
    user_confirmation_router,
    extract_reservation_details_node,
    finalize_reservation_node,
    missed_reservation_details_router,
    intent_router,
    interpret_user_confirmation_node,
    qa_system_rag_output_node,
    qa_system_rag_input_node,
    request_reservation_confirmation_node,
    validate_reservation_details
)
from chatbot.graph.states import GraphState


def build_graph() -> StateGraph:

    g = StateGraph(GraphState)

    # Guardrail
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("blocked_response", blocked_response_node)
    g.add_node("output_guardrail", output_guardrail_node)

    # root
    g.add_node("classify_intent", classify_user_intent_node)

    # QA nodes
    g.add_node("information_request", qa_system_rag_input_node)
    g.add_node("information_response", qa_system_rag_output_node)

    # unknown
    g.add_node("unknown", unknown_node)

    # Reservation nodes
    g.add_node("extract_details", extract_reservation_details_node)
    g.add_node("validate_details", validate_reservation_details)
    g.add_node("ask_missed_details", ask_missed_reservation_details_node)
    g.add_node("request_user_confirmation", request_reservation_confirmation_node)
    g.add_node("interpret_user_confirmation", interpret_user_confirmation_node)
    g.add_node("finalize_reservation", finalize_reservation_node)
    g.add_node("cancel_reservation", cancel_inprogress_reservation_node)


    # Guardrail
    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges(
        "input_guardrail",
        after_guardrail_router,
        {
            "unknown": "unknown",
            "blocked_response": "blocked_response",
            "classify_intent": "classify_intent",
            "extract_details": "extract_details",
            "interpret_user_confirmation": "interpret_user_confirmation",
        },
    )

    # unknown
    g.add_edge("unknown", END)

    # Root router to define communication direction
    g.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "unknown": "unknown",
            "information_request": "information_request",
            "extract_details": "extract_details",
        },
    )

    # QA edges
    g.add_edge("information_request", "information_response")
    g.add_edge("information_response", "output_guardrail")

    # Reservation edges
    g.add_edge("extract_details", "validate_details")
    # Collect reservation details
    g.add_conditional_edges(
        "validate_details",
        missed_reservation_details_router,
        {
            "ask_missed_details": "ask_missed_details",
            "request_user_confirmation": "request_user_confirmation",
        },
    )
    # Final reservation status
    g.add_conditional_edges(
        "interpret_user_confirmation",
        user_confirmation_router,
        {
            "finalize_reservation": "finalize_reservation",
            "cancel_reservation": "cancel_reservation",
            "extract_details": "extract_details",
            "request_user_confirmation": "request_user_confirmation",
        },
    )
    g.add_edge("ask_missed_details", "output_guardrail")
    g.add_edge("request_user_confirmation", "output_guardrail")
    g.add_edge("finalize_reservation", "output_guardrail")
    g.add_edge("cancel_reservation", "output_guardrail")
    g.add_edge("output_guardrail", END)
    g.add_edge("blocked_response", END)

    state = g.compile(checkpointer=InMemorySaver())
    png_data = state.get_graph().draw_mermaid_png()

    with open("langgraph.png", "wb") as f:
        f.write(png_data)

    return state

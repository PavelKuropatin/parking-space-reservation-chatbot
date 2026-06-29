from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from chatbot.graph.nodes import (
    after_input_guardrail_router,
    blocked_response_node,
    classify_user_intent_node,
    input_guardrail_node,
    output_guardrail_node,
    unknown_node,
    extract_reservation_details_node,
    after_classify_intent_router,
    qa_system_rag_output_node,
    qa_system_rag_input_node
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
    g.add_node("reservation", extract_reservation_details_node)


    # Guardrail
    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges(
        "input_guardrail",
        after_input_guardrail_router,
        {
            "blocked_response": "blocked_response",
            "classify_intent": "classify_intent",
            "reservation": "reservation",
        },
    )

    # unknown
    g.add_edge("unknown", "output_guardrail")

    # Root router to define communication direction
    g.add_conditional_edges(
        "classify_intent",
        after_classify_intent_router,
        {
            "unknown": "unknown",
            "information_request": "information_request",
            "reservation": "reservation",
        },
    )

    # QA edges
    g.add_edge("information_request", "information_response")
    g.add_edge("information_response", "output_guardrail")

    # Reservation edges
    g.add_edge("reservation", "output_guardrail")
    g.add_edge("blocked_response", "output_guardrail")

    # output
    g.add_edge("output_guardrail", END)

    state = g.compile(checkpointer=InMemorySaver())
    png_data = state.get_graph().draw_mermaid_png()

    with open("langgraph.png", "wb") as f:
        f.write(png_data)

    return state

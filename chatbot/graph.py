from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph.state import CompiledStateGraph
from chatbot.nodes import (
    after_input_guardrail_router,
    after_reservation_router,
    blocked_response_node,
    classify_user_intent_node,
    input_guardrail_node,
    output_guardrail_node,
    process_admin_response_node,
    request_admin_approval_node,
    unknown_node,
    create_reservation_node,
    save_reservation_node,
    after_classify_intent_router,
    qa_system_rag_node
)
from chatbot.states import GraphState


def build_graph(checkpointer: PostgresSaver) -> CompiledStateGraph:

    g = StateGraph(GraphState)

    # Guardrail
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("blocked_response", blocked_response_node)
    g.add_node("output_guardrail", output_guardrail_node)

    # root
    g.add_node("classify_intent", classify_user_intent_node)

    # QA nodes
    g.add_node("information", qa_system_rag_node)

    # unknown
    g.add_node("_unknown", unknown_node)

    # Reservation nodes
    g.add_node("reservation", create_reservation_node)
    g.add_node("save_reservation", save_reservation_node)
    g.add_node("request_admin_approval", request_admin_approval_node)
    g.add_node("process_admin_response", process_admin_response_node)


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
    g.add_edge("blocked_response", "output_guardrail")

    # Root router to define communication direction
    g.add_conditional_edges(
        "classify_intent",
        after_classify_intent_router,
        {
            "_unknown": "_unknown",
            "information": "information",
            "reservation": "reservation",
        },
    )
    # unknown
    g.add_edge("_unknown", "output_guardrail")

    # QA edges
    g.add_edge("information", "output_guardrail")

    # Reservation edges
    g.add_edge("reservation", "output_guardrail")
    g.add_conditional_edges(
        "reservation",
        after_reservation_router,
        {
            "save_reservation": "save_reservation",
            "output_guardrail": "output_guardrail",
        }
    )
    # ignore interrupted node rerun
    g.add_edge("save_reservation", "request_admin_approval")
    g.add_edge("request_admin_approval", "process_admin_response")
    g.add_edge("process_admin_response", "output_guardrail")

    # output
    g.add_edge("output_guardrail", END)

    state = g.compile(checkpointer=checkpointer)
    return state

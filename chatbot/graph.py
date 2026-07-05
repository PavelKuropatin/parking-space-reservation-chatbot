import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph.state import CompiledStateGraph
from chatbot.nodes import (
    after_input_guardrail_router,
    after_request_admin_approval_router,
    after_reservation_router,
    blocked_response_node,
    classify_user_intent_node,
    input_guardrail_node,
    output_guardrail_node,
    process_admin_rejection_node,
    request_admin_approval_node,
    unknown_node,
    create_reservation_node,
    after_classify_intent_router,
    qa_system_rag_node,
)
from chatbot.settings import get_settings
from chatbot.states import GraphState, ReservationPhase
from chatbot.logging import logger


async def get_mcp_tools() -> dict:
    settings = get_settings()
    client = MultiServerMCPClient(
        {
            "parking": {
                "transport": "http",
                "url": settings.mcp_url,
                "headers": {"Authorization": f"Bearer {settings.mcp_client_token}"},
            }
        }
    )
    return {t.name: t for t in await client.get_tools()}


async def build_graph(checkpointer: PostgresSaver) -> CompiledStateGraph:

    mcp_tools = await get_mcp_tools()

    # FIXME convert to llm call + tool_choice, the same for rag and reservation
    async def process_admin_approval_node(state: GraphState) -> dict:
        reservation_details = state.get("reservation_details")
        submit_reservation_tool = mcp_tools["submit_reservation"]

        try:
            tool_messages = await submit_reservation_tool.ainvoke(
                {"reservation": reservation_details}
            )
            response = json.loads(tool_messages[0]['text'])
            is_error = response["is_error"]
            if is_error:
                raise Exception(f"MCP: {response["error_message"]}")

            reservation_id = response["reservation_id"]
            return {
                "route": None,
                "reservation_phase": ReservationPhase.COMPLETED,
                "reservation_id": reservation_id,
                "reservation_details": {},
                "ai_message": f"Your reservaton #{reservation_id} was approved.",
            }
        except Exception as e:
            logger.error(
                "Failed to create reservation: %s",
                e,
            )
            return {
                "route": None,
                "reservation_phase": None,
                "reservation_details": {},
                "ai_message": "Failed to create reservation. Please try again later.",
            }

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
    g.add_node("request_admin_approval", request_admin_approval_node)
    g.add_node("process_admin_rejection", process_admin_rejection_node)
    g.add_node("process_admin_approval", process_admin_approval_node)

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
            "request_admin_approval": "request_admin_approval",
            "output_guardrail": "output_guardrail",
        },
    )

    # ignore interrupted node rerun
    g.add_conditional_edges(
        "request_admin_approval",
        after_request_admin_approval_router,
        {
            "process_admin_rejection": "process_admin_rejection",
            "process_admin_approval": "process_admin_approval",
        },
    )
    g.add_edge("process_admin_rejection", "output_guardrail")
    g.add_edge("process_admin_approval", "output_guardrail")

    # output
    g.add_edge("output_guardrail", END)

    state = g.compile(checkpointer=checkpointer)
    return state

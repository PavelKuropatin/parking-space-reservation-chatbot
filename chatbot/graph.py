from langchain.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from chatbot.nodes import (
    after_reservation_router,
    after_data_recording_router,
    after_input_guardrail_router,
    after_request_admin_approval_router,
    create_reservation_node,
    ask_missed_reservation_details_node,
    blocked_response_node,
    classify_user_input_node,
    confirm_reservation_details_node,
    submit_reservation_admin_node,
    finalize_reservation_client_node,
    input_guardrail_node,
    output_guardrail_node,
    process_admin_rejection_node,
    request_admin_approval_node,
    unknown_node,
    after_classify_user_input_router,
    qa_system_rag_node,
)
from chatbot.prompts import DATA_RECORDING_SYSTEM_PROMPT
from chatbot.settings import get_settings
from chatbot.states import GraphState
from chatbot.utils.graph_utils import get_llm


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
    mcp_tools = list(mcp_tools.values())

    data_recording_llm = get_llm(temperature=0.3).bind_tools(
        mcp_tools, tool_choice="auto"
    )
    data_recording_tools = ToolNode(mcp_tools, messages_key="data_recording_messages")

    async def data_recording_agent(state: GraphState) -> dict:
        response = data_recording_llm.invoke(
            [
                SystemMessage(DATA_RECORDING_SYSTEM_PROMPT),
                *state["data_recording_messages"],
            ]
        )
        return {"data_recording_messages": [response]}

    g = StateGraph(GraphState)

    # Guardrail
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("blocked_response", blocked_response_node)
    g.add_node("output_guardrail", output_guardrail_node)

    # root
    g.add_node("classify_user_input", classify_user_input_node)
    g.add_node("information", qa_system_rag_node)
    g.add_node("_unknown", unknown_node)
    g.add_node("reservation", create_reservation_node)

    # Reservation nodes (client)
    g.add_node("ask_missed_reservation_details", ask_missed_reservation_details_node)
    g.add_node("confirm_reservation_details", confirm_reservation_details_node)
    g.add_node("finalize_reservation_client", finalize_reservation_client_node)

    # Reservation nodes (admin)
    g.add_node("request_admin_approval", request_admin_approval_node)
    g.add_node("process_admin_rejection", process_admin_rejection_node)

    # MCP
    g.add_node("data_recording_agent", data_recording_agent)
    g.add_node("data_recording_tools", data_recording_tools)
    g.add_node("_submit_reservation_admin", submit_reservation_admin_node)

    # Flow
    # Guardrail
    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges(
        "input_guardrail",
        after_input_guardrail_router,
        ["blocked_response", "classify_user_input", "reservation"],
    )
    g.add_edge("blocked_response", "output_guardrail")

    # Root router to define communication direction
    g.add_conditional_edges(
        "classify_user_input",
        after_classify_user_input_router,
        ["_unknown", "information", "reservation"],
    )

    # reservation
    g.add_conditional_edges(
        "reservation",
        after_reservation_router,
        [
            "ask_missed_reservation_details",
            "confirm_reservation_details",
            "finalize_reservation_client",
        ],
    )

    # unknown
    g.add_edge("_unknown", "output_guardrail")

    # QA edges
    g.add_edge("information", "output_guardrail")

    # Reservation edges
    # g.add_edge("reservation", "output_guardrail")
    g.add_edge("ask_missed_reservation_details", "output_guardrail")
    g.add_edge("confirm_reservation_details", "output_guardrail")

    g.add_edge("finalize_reservation_client", "request_admin_approval")

    # ignore interrupted node rerun
    g.add_conditional_edges(
        "request_admin_approval",
        after_request_admin_approval_router,
        ["process_admin_rejection", "data_recording_agent"],
    )
    # mcp
    g.add_conditional_edges(
        "data_recording_agent",
        after_data_recording_router,
        ["data_recording_tools", "_submit_reservation_admin"],
    )
    g.add_edge("data_recording_tools", "data_recording_agent")

    g.add_edge("_submit_reservation_admin", "output_guardrail")
    g.add_edge("process_admin_rejection", "output_guardrail")

    # output
    g.add_edge("output_guardrail", END)
    state = g.compile(checkpointer=checkpointer)

    png_data = state.get_graph().draw_mermaid_png()
    with open("langgraph.png", "wb") as f:
        f.write(png_data)

    return state

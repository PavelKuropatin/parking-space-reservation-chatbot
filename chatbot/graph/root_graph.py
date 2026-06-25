


from typing import Annotated, TypedDict

from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from chatbot.database.sql_store import ParkingData
from chatbot.settings import Settings
from chatbot.utils.sql_store_utils import make_sql_store_tools

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    block: bool

# core
def get_llm_model(settings: Settings):
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_llm_url,
    )


def build_assistant_graph(settings: Settings, parking_data_db: ParkingData, parking_info_vs: WeaviateVectorStore) -> StateGraph:

    tools = make_sql_store_tools(parking_data_db)
    llm =  get_llm_model(settings)
    llm_with_tools = llm.bind_tools(tools)


    # nodes
    def input_pii_guard(state: ChatState) -> dict:
        last_message = state["messages"][-1]
        # todo pii check
        return {"block" : False }
    
    def agent(state: ChatState) -> dict:
        # todo separate graph for user infor retrieval
        mesasges = state["messages"]
        return { "messages": [llm_with_tools.invoke(mesasges)] }
    
    def output_pii_guard(state: ChatState) -> dict:
        # todo pii check ?
        last_message = state["messages"][-1]
        return {"block" : False }
    
    # routing
    def after_input(state: ChatState) -> str:
        if state.get("block"):
            return "block_thread" 
        
        # todo llm call to categorize message (get_info/book_place/other)
        return "agent"

    def after_agent(state: ChatState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "output_pii_guard"

    graph = StateGraph(ChatState)
    graph.add_node("input_pii_guard", input_pii_guard)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output_pii_guard", output_pii_guard)

    graph.add_edge(START, "input_pii_guard")
    graph.add_conditional_edges("input_pii_guard", after_input, {"block_thread": END, "agent": "agent"})
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "output_pii_guard": "output_pii_guard"})
    graph.add_edge("tools", "agent")
    graph.add_edge("output_pii_guard", END)

    # todo PostgreSQL saver
    return graph.compile(checkpointer=InMemorySaver())
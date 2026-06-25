from langchain.messages import HumanMessage
from langchain_weaviate import WeaviateVectorStore

from chatbot.database.sql_store import ParkingData
from chatbot.graph.root_graph import ChatState, build_assistant_graph
from chatbot.settings import Settings


class ParkingAssistant:

    def __init__(
        self,
        settings: Settings,
        parking_data_db: ParkingData,
        parking_info_vs: WeaviateVectorStore,
    ):
        self.__graph = build_assistant_graph(settings, parking_data_db, parking_info_vs)

    def chat(self, text: str) -> ChatState:
        # todo ?
        return self.__graph.invoke(
            {"messages": [HumanMessage(content=text)], "block": False},
            config={"configurable": {"thread_id": "default"}},
        )

from langchain.messages import HumanMessage

from chatbot.graph.client_graph import GraphState, build_graph


# pylint: disable=too-few-public-methods
class ParkingAssistant:

    def __init__(self, thread_id: str):
        self.__graph = build_graph()
        self.__config = {"configurable": {"thread_id": thread_id}}

    def chat(self, user_input: str) -> GraphState:
        response = self.__graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=self.__config
        )
        return response

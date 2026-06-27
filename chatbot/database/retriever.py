from chatbot.settings import get_settings
from chatbot.utils.singleton import Singleton
from chatbot.utils.weaviate_utils import get_weaviate_client


class ParkingInformationRetriever(metaclass=Singleton):

    def __init__(self):
        settings = get_settings()
        self.__client = get_weaviate_client()
        self.__top_k = settings.rag_top_k
        self.__collection = settings.weaviate_collection

    @property
    def top_k(self) -> int:
        return self.__top_k

    def query(self, query: str):
        return self.__client.collections.get(self.__collection).query.hybrid(
            query=query, limit=self.__top_k
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.__client.close()
        finally:
            pass

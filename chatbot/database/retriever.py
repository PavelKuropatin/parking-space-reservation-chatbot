from chatbot.settings import Settings, get_settings
from chatbot.utils.weaviate_utils import get_weaviate_client


class ParkingInformationRetriever:

    def __init__(self, settings: Settings):
        self.__client = get_weaviate_client(settings)
        self.__top_k = settings.rag_top_k
        self.__collection = settings.weaviate_collection

    @property
    def top_k(self) -> int:
        return self.__top_k

    def query(self, query: str, top_k: int = None):
        if not top_k:
            top_k = self.__top_k
        return self.__client.collections.get(self.__collection).query.hybrid(
            query=query, limit=top_k
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.__client.close()
        finally:
            pass


__PARKING_INFO_RETRIEVER: ParkingInformationRetriever = None


def get_parking_info_retriever() -> ParkingInformationRetriever:
    global __PARKING_INFO_RETRIEVER
    if __PARKING_INFO_RETRIEVER is None:
        __PARKING_INFO_RETRIEVER = ParkingInformationRetriever(get_settings())
    return __PARKING_INFO_RETRIEVER

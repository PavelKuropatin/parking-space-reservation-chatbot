from chatbot.utils.settings import Settings
from chatbot.utils.weaviate_utils import get_weaviate_client, get_weaviate_vector_store


class ParkingInformationRetriever:

    def __init__(self, settings: Settings):
        self.__client = get_weaviate_client(settings)
        self.__vestor_store = get_weaviate_vector_store(self.__client, settings)
        self.__top_k = settings.rag_top_k

    def query(self, query: str):
        return self.__vestor_store.similarity_search(query=query, k=self.__top_k)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        try:
            self.__client.close()
        finally:
            pass
    
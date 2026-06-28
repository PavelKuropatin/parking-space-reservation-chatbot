import weaviate
from weaviate import WeaviateClient
from chatbot.settings import Settings


def get_weaviate_client(settings: Settings) -> WeaviateClient:
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )

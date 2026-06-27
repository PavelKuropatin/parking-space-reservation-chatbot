import weaviate
from weaviate import WeaviateClient
from chatbot.settings import get_settings


def get_weaviate_client() -> WeaviateClient:
    settings = get_settings()
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )

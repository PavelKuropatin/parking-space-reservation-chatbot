import weaviate
from weaviate import WeaviateClient
from langchain.embeddings import init_embeddings
from langchain_weaviate import WeaviateVectorStore
from chatbot.settings import Settings


def get_weaviate_client(settings: Settings) -> WeaviateClient:
    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def get_weaviate_vector_store(
    client: WeaviateClient, settings: Settings
) -> WeaviateVectorStore:
    client = get_weaviate_client(settings)
    embeddings = init_embeddings(
        model=settings.openai_embeddings_model,
        openai_api_key=settings.openai_api_key,
    )
    return WeaviateVectorStore(
        client=client,
        index_name=settings.weaviate_collection,
        text_key="content",
        embedding=embeddings,
        attributes=["content_category"],
    )

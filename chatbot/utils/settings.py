from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    openai_api_key: str
    openai_embeddings_model: str

    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    weaviate_collection: str
    weaviate_init_data_path: str

    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int
    
    model_config =  ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings():
    return Settings()

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    llm_model_provider: str
    llm_model_name: str
    llm_url: str
    llm_api_key: str

    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    weaviate_collection: str
    weaviate_init_data_path: str

    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_pswd: str
    postgres_db: str
    postgres_pool_min_size: int
    postgres_pool_max_size: int

    checkpointer_host: str
    checkpointer_port: int
    checkpointer_user: str
    checkpointer_pswd: str
    checkpointer_db: str

    notification_path: str

    mcp_url: str
    mcp_client_token: str

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings():
    return Settings()

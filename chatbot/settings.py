from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    openai_llm_model: str
    openai_llm_url: str
    openai_llm_api_key: str

    openai_embeddings_api_key: str
    openai_embeddings_model: str

    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    weaviate_collection: str
    weaviate_init_data_path: str

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_pswd: str
    postgres_db: str
    postgres_pool_min_size: int
    postgres_pool_max_size: int

    rag_top_k: int
    rag_chunk_size: int
    rag_chunk_overlap: int

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

@lru_cache
def get_settings():
    return Settings()

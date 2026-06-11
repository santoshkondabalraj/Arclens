from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Google / Gemini
    google_api_key: str = ""
    gemini_embedding_model: str = "models/text-embedding-004"
    gemini_flash_model: str = "gemini-2.5-flash"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "yt-rag-hybrid"
    pinecone_environment: str = "us-east-1-aws"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "yt-rag"

    # Retrieval thresholds
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_n: int = 10
    min_rerank_score: float = -10.0

    # Chunking (tokens)
    body_chunk_size: int = 600
    body_chunk_overlap: int = 150
    meta_chunk_size: int = 192

    # Freshness
    staleness_days: int = 30

    # Storage paths
    bm25_index_dir: str = "data/bm25_indexes"
    ingestion_meta_dir: str = "data/ingestion_meta"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

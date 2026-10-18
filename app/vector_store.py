"""pgvector store (its own Postgres container, separate from the compensation DB)."""

from functools import lru_cache

from langchain_postgres import PGVector

from app.config import get_settings
from app.llm import get_embeddings


@lru_cache
def get_vector_store() -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.vector_collection,
        connection=settings.vector_database_url,
        use_jsonb=True,
    )

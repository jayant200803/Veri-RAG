"""Central configuration. All values overridable via .env"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider selection ---
    llm_provider: str = "gemini"           # gemini | groq | ollama

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "verirag_docs"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings (local, free) ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384                    # bge-small-en-v1.5

    # --- Agent thresholds (defend these numbers on stage) ---
    confidence_answer_threshold: float = 0.7
    confidence_abstain_threshold: float = 0.4
    max_requery_attempts: int = 2
    retrieval_top_k: int = 6

    # --- Chunking ---
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- Resilience ---
    llm_timeout_seconds: int = 45
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

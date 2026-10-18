"""All configuration comes from environment variables (or a local .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Claude chat models (ANTHROPIC_API_KEY is read directly from the environment by LangChain).
    llm_fast_model: str = "claude-haiku-5-5"  # classify + both answer nodes
    llm_sql_model: str = "claude-haiku-5-5"  # generate_sql
    # Gemini fallback chat model (GOOGLE_API_KEY, also used for embeddings).
    llm_model: str = ""  # empty -> gemini-2.5-flash

    # Existing compensation database (connect with a READ-ONLY user).
    database_url: str
    # Optional comma-separated allowlist of tables the LLM may see, e.g. "employees,salaries".
    # Empty means every table in the default schema.
    comp_tables: str = ""

    # Separate pgvector database for document embeddings.
    vector_database_url: str
    vector_collection: str = "hr_policies"

    # JWTs issued by the compensation app (HS256, shared secret).
    jwt_secret: str
    jwt_role_claim: str = "role"
    hr_role_value: str = "HR"

    @property
    def table_list(self) -> list[str]:
        return [t.strip() for t in self.comp_tables.split(",") if t.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

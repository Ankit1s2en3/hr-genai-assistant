"""Read-only access to the existing compensation database."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def describe_schema(engine: Engine, tables: list[str] | None = None) -> str:
    """Build a compact schema description for the LLM, e.g. `employees(id INTEGER, name VARCHAR)`."""
    inspector = inspect(engine)
    lines = []
    for table in tables or inspector.get_table_names():
        columns = ", ".join(f"{c['name']} {c['type']}" for c in inspector.get_columns(table))
        lines.append(f"{table}({columns})")
    return "\n".join(lines)


def run_select(engine: Engine, sql: str) -> list[dict]:
    """Run an already-validated SELECT inside a read-only transaction with a timeout."""
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = 5000"))  # milliseconds
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]
        # Leaving the block rolls the transaction back; nothing is ever committed.

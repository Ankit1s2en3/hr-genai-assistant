"""Checks that LLM-generated SQL is a single, read-only SELECT before we run it.

We parse the SQL into a syntax tree with sqlglot instead of using regexes, so tricks like
`SELECT 1; DROP TABLE x` or a DELETE hidden inside a CTE are caught reliably.
This is defence in depth: the database user is also read-only.
"""

import sqlglot
from sqlglot import exp

MAX_ROWS = 100

# Statement types that must never appear anywhere in the query tree.
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Merge,
    exp.TruncateTable,
    exp.Command,  # anything sqlglot doesn't model (GRANT, COPY, VACUUM, ...)
)


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Return a safe version of `sql` (with a LIMIT), or raise UnsafeSQLError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed")

    query = statements[0]
    if not isinstance(query, exp.Select):
        raise UnsafeSQLError("Only SELECT statements are allowed")

    if query.find(*FORBIDDEN_NODES):
        raise UnsafeSQLError("Query contains a forbidden operation")

    if query.args.get("into"):  # SELECT ... INTO new_table creates a table
        raise UnsafeSQLError("SELECT INTO is not allowed")

    if query.args.get("limit") is None:
        query = query.limit(MAX_ROWS)

    return query.sql(dialect="postgres")

import pytest

from app.sql_guard import UnsafeSQLError, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM salaries",
        "UPDATE salaries SET amount = 1000000 WHERE employee_id = 7",
        "SELECT * FROM employees; DROP TABLE employees",
        "SELECT 1; SELECT 2",
        "DROP TABLE employees",
        "ALTER TABLE employees ADD COLUMN hacked BOOLEAN",
        "INSERT INTO employees (name) VALUES ('x')",
        "WITH gone AS (DELETE FROM salaries RETURNING *) SELECT * FROM gone",
        "SELECT * INTO stolen FROM salaries",
        "GRANT ALL ON salaries TO public",
        "this is not sql",
    ],
)
def test_rejects_unsafe_sql(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_adds_limit_when_missing():
    result = validate_sql("SELECT name FROM employees WHERE department = 'Sales'")
    assert result.endswith("LIMIT 100")


def test_keeps_existing_limit():
    result = validate_sql("SELECT name FROM employees LIMIT 5")
    assert result.endswith("LIMIT 5")
    assert "LIMIT 100" not in result


def test_allows_joins_and_aggregates():
    sql = (
        "SELECT e.department, AVG(s.amount) FROM employees e "
        "JOIN salaries s ON s.employee_id = e.id GROUP BY e.department"
    )
    assert validate_sql(sql).startswith("SELECT")

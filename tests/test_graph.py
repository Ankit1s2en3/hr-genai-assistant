"""Graph tests with a mocked LLM, retriever and SQL runner. No real API or DB calls."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.graph import NOT_ALLOWED, RouteDecision, SQLQuery, build_graph


HAIKU = "claude-haiku-5-5"


def make_llm(structured_outputs: list, answer: str = "mocked answer", model: str = HAIKU) -> MagicMock:
    """A fake chat model that reports itself as `model` in response_metadata.

    `structured_outputs` are returned in order by `llm.with_structured_output(...).invoke()`,
    in the include_raw=True shape; plain `llm.invoke()` returns an AIMessage with `answer`.
    """
    metadata = {"model_name": model}
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        {"raw": AIMessage(content="", response_metadata=metadata), "parsed": out, "parsing_error": None}
        for out in structured_outputs
    ]
    llm.invoke.return_value = AIMessage(content=answer, response_metadata=metadata)
    return llm


def make_graph(llm, retriever=None, run_sql=None, sql_llm=None):
    """`llm` plays the classify and answer models; `sql_llm` defaults to the same fake."""
    return build_graph(
        classify_llm=llm,
        sql_llm=sql_llm or llm,
        answer_llm=llm,
        retriever=retriever or MagicMock(),
        run_sql=run_sql or MagicMock(),
        schema="employees(id INTEGER, name VARCHAR, department VARCHAR)",
    )


def test_classify_routes_policy_question_to_docs():
    llm = make_llm([RouteDecision(route="docs")], answer="Employees get 25 days of annual leave.")
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(page_content="Annual leave is 25 days.", metadata={"filename": "leave.pdf", "chunk": 3})
    ]
    run_sql = MagicMock()

    result = make_graph(llm, retriever, run_sql).invoke(
        {"question": "How many vacation days do I get?", "user_role": "Employee"}
    )

    assert result["route"] == "docs"
    assert result["answer"] == "Employees get 25 days of annual leave."
    assert result["sources"] == [{"filename": "leave.pdf", "chunk": 3}]
    assert result["models"] == {"classify": HAIKU, "answer": HAIKU}
    llm.with_structured_output.assert_called_with(RouteDecision, include_raw=True)
    retriever.invoke.assert_called_once_with("How many vacation days do I get?")
    run_sql.assert_not_called()


def test_docs_route_says_i_dont_know_without_context():
    llm = make_llm([RouteDecision(route="docs")])
    retriever = MagicMock()
    retriever.invoke.return_value = []

    result = make_graph(llm, retriever).invoke({"question": "What is the dress code?", "user_role": "HR"})

    assert "I don't know" in result["answer"]
    llm.invoke.assert_not_called()  # no LLM call when there is nothing to answer from


def test_non_hr_user_is_refused_on_data_route():
    llm = make_llm([RouteDecision(route="data")])
    run_sql = MagicMock()

    result = make_graph(llm, run_sql=run_sql).invoke(
        {"question": "What is the average salary in Sales?", "user_role": "Employee"}
    )

    assert result["route"] == "data"
    assert result["answer"] == NOT_ALLOWED
    assert "sql" not in result
    assert result["models"] == {"classify": HAIKU}  # the refusal itself needs no LLM
    run_sql.assert_not_called()


def test_hr_user_gets_answer_from_database():
    llm = make_llm([RouteDecision(route="data")], answer="There are 42 employees.")
    # Pretend the SQL call fell back to Gemini, to check each step reports its own model.
    sql_llm = make_llm([SQLQuery(sql="SELECT COUNT(*) AS n FROM employees")], model="gemini-2.5-flash")
    run_sql = MagicMock(return_value=[{"n": 42}])

    result = make_graph(llm, run_sql=run_sql, sql_llm=sql_llm).invoke(
        {"question": "How many employees do we have?", "user_role": "HR"}
    )

    assert result["answer"] == "There are 42 employees."
    assert result["sql"].endswith("LIMIT 100")  # validator added the limit
    run_sql.assert_called_once_with(result["sql"])
    sql_llm.with_structured_output.assert_called_once_with(SQLQuery, include_raw=True)
    assert result["models"] == {"classify": HAIKU, "generate_sql": "gemini-2.5-flash", "answer": HAIKU}


def test_unsafe_generated_sql_is_never_executed():
    llm = make_llm([RouteDecision(route="data"), SQLQuery(sql="DELETE FROM salaries")])
    run_sql = MagicMock()

    result = make_graph(llm, run_sql=run_sql).invoke(
        {"question": "Remove all salaries", "user_role": "HR"}
    )

    assert "safe" in result["answer"]
    run_sql.assert_not_called()

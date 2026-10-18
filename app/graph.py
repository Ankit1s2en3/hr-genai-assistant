"""The LangGraph workflow.

    classify ──docs──> retrieve -> answer_from_docs -> END
        │
        ├──data (not HR)──> refuse -> END
        │
        └──data (HR)──> generate_sql -> validate_sql -> run_sql -> answer_from_rows -> END
                                             │             │
                                             └── error ────┴──> END

Dependencies (LLMs, retriever, SQL runner, schema) are passed in, so tests can use fakes.
"""

import json
import logging
import operator
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.llm import model_name
from app.sql_guard import UnsafeSQLError, validate_sql

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    user_role: str
    route: Literal["docs", "data"]
    context: str
    sql: str
    rows: list[dict]
    answer: str
    sources: list[dict]
    error: str
    # Which model answered each step, e.g. {"classify": "claude-haiku-5-5"}.
    # operator.or_ merges each node's update into the dict instead of replacing it.
    models: Annotated[dict[str, str], operator.or_]


# --- Structured LLM outputs -------------------------------------------------------------


class RouteDecision(BaseModel):
    """Which path should answer the question."""

    route: Literal["docs", "data"] = Field(
        description="'docs' for questions about HR policies, rules, benefits or procedures. "
        "'data' for questions needing numbers about employees, salaries, departments or headcount."
    )


class SQLQuery(BaseModel):
    """A single read-only PostgreSQL query."""

    sql: str = Field(description="Exactly one PostgreSQL SELECT statement.")


# --- Prompts and fixed answers -----------------------------------------------------------

CLASSIFY_PROMPT = (
    "You route questions for an HR assistant. Choose 'docs' if the question is about company "
    "policies, rules, benefits or procedures. Choose 'data' if it asks for facts or numbers "
    "about employees or salaries that must be looked up in the database."
)

DOCS_PROMPT = """You are an HR assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "{i_dont_know}"

Context:
{context}"""

SQL_PROMPT = """You write PostgreSQL queries for an HR compensation database.
Rules:
- Return exactly ONE SELECT statement. Never modify data.
- Only use the tables and columns in the schema below.
- Name the columns you need instead of using SELECT *.

Schema:
{schema}"""

DATA_ANSWER_PROMPT = """You are an HR assistant. Answer the user's question in plain English
using only the query result below. If there are no rows, say no matching data was found.

SQL: {sql}
Rows: {rows}"""

I_DONT_KNOW = "I don't know based on the available policy documents."
NOT_ALLOWED = (
    "Sorry, only HR users can ask about employee or salary data. "
    "I'm happy to help with questions about HR policies, though."
)
UNSAFE_SQL = "I couldn't build a safe read-only query for that question. Please try rephrasing it."
QUERY_FAILED = "I couldn't run the database query for that question. Please try rephrasing it."


def invoke_structured(llm: Runnable, schema: type[BaseModel], messages: list):
    """Call `llm` for a `schema` object, and also return which model answered.

    include_raw=True keeps the raw AIMessage (and its response_metadata) next to the parsed object.
    """
    result = llm.with_structured_output(schema, include_raw=True).invoke(messages)
    if result["parsed"] is None:
        raise ValueError(f"Model did not return a valid {schema.__name__}: {result['parsing_error']}")
    return result["parsed"], model_name(result["raw"])


def build_graph(
    classify_llm: Runnable,
    sql_llm: Runnable,
    answer_llm: Runnable,
    retriever: BaseRetriever,
    run_sql: Callable[[str], list[dict]],
    schema: str,
    hr_role: str = "HR",
):
    # --- Nodes: each takes the state and returns only the fields it changes. ---

    def classify(state: AgentState) -> AgentState:
        decision, model = invoke_structured(
            classify_llm, RouteDecision, [("system", CLASSIFY_PROMPT), ("human", state["question"])]
        )
        return {"route": decision.route, "models": {"classify": model}}

    def retrieve(state: AgentState) -> AgentState:
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(
            f"[{d.metadata.get('filename')} #{d.metadata.get('chunk')}]\n{d.page_content}"
            for d in docs
        )
        sources = [
            {"filename": d.metadata.get("filename"), "chunk": d.metadata.get("chunk")} for d in docs
        ]
        return {"context": context, "sources": sources}

    def answer_from_docs(state: AgentState) -> AgentState:
        if not state.get("context"):
            return {"answer": I_DONT_KNOW}
        prompt = DOCS_PROMPT.format(i_dont_know=I_DONT_KNOW, context=state["context"])
        reply = answer_llm.invoke([("system", prompt), ("human", state["question"])])
        return {"answer": reply.text, "models": {"answer": model_name(reply)}}

    def refuse(state: AgentState) -> AgentState:
        return {"answer": NOT_ALLOWED}

    def generate_sql(state: AgentState) -> AgentState:
        query, model = invoke_structured(
            sql_llm, SQLQuery, [("system", SQL_PROMPT.format(schema=schema)), ("human", state["question"])]
        )
        return {"sql": query.sql, "models": {"generate_sql": model}}

    def validate(state: AgentState) -> AgentState:
        try:
            return {"sql": validate_sql(state["sql"])}
        except UnsafeSQLError as e:
            logger.warning("Rejected SQL %r: %s", state["sql"], e)
            return {"error": str(e), "answer": UNSAFE_SQL}

    def run_query(state: AgentState) -> AgentState:
        try:
            return {"rows": run_sql(state["sql"])}
        except Exception as e:
            logger.exception("SQL execution failed")
            return {"error": str(e), "answer": QUERY_FAILED}

    def answer_from_rows(state: AgentState) -> AgentState:
        rows = json.dumps(state["rows"], default=str)  # default=str handles dates and Decimals
        prompt = DATA_ANSWER_PROMPT.format(sql=state["sql"], rows=rows)
        reply = answer_llm.invoke([("system", prompt), ("human", state["question"])])
        return {"answer": reply.text, "models": {"answer": model_name(reply)}}

    # --- Conditional edges ---

    def route_after_classify(state: AgentState) -> str:
        if state["route"] == "docs":
            return "retrieve"
        if state.get("user_role") != hr_role:
            return "refuse"
        return "generate_sql"

    def continue_unless_error(state: AgentState) -> str:
        return "stop" if state.get("error") else "continue"

    # --- Wire it up ---

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer_from_docs", answer_from_docs)
    graph.add_node("refuse", refuse)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate)
    graph.add_node("run_sql", run_query)
    graph.add_node("answer_from_rows", answer_from_rows)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, ["retrieve", "refuse", "generate_sql"])

    graph.add_edge("retrieve", "answer_from_docs")
    graph.add_edge("answer_from_docs", END)

    graph.add_edge("refuse", END)

    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", continue_unless_error, {"continue": "run_sql", "stop": END})
    graph.add_conditional_edges("run_sql", continue_unless_error, {"continue": "answer_from_rows", "stop": END})
    graph.add_edge("answer_from_rows", END)

    return graph.compile()

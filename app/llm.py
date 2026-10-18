"""Factories for the chat model (Claude, with Gemini as fallback) and the Gemini embedding model.

API keys are read from the environment by the LangChain clients:
ANTHROPIC_API_KEY for Claude, GOOGLE_API_KEY for Gemini.
"""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 768

# Output caps keep costs predictable.
SHORT_MAX_TOKENS = 256  # classify + SQL: tiny structured outputs
ANSWER_MAX_TOKENS = 800  # plain-English answers

MAX_RETRIES = 2
TIMEOUT_SECONDS = 30


class LogFallbackToGemini(BaseCallbackHandler):
    """Fires when a Claude call fails for good (after the SDK's own retries), right before
    `.with_fallbacks` hands the same request to Gemini."""

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        logger.warning("Claude call failed, falling back to Gemini: %r", error)


def model_name(message: BaseMessage) -> str:
    """Which model produced this response, e.g. 'claude-haiku-5-5' or 'gemini-2.5-flash'."""
    return message.response_metadata.get("model_name", "unknown")


def get_chat_model(model: str, max_tokens: int) -> Runnable:
    """Claude first; if the call fails (outage, rate limit, timeout), Gemini answers instead."""
    claude = ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT_SECONDS,
        callbacks=[LogFallbackToGemini()],
        # No temperature: Claude Haiku 5.5 rejects non-default sampling values.
        # Thinking off: these are short, simple calls, and thinking tokens would count
        # against max_tokens. (Sonnet/Opus 5.5 reject "disabled"; see README.)
        thinking={"type": "disabled"},
    )
    gemini = ChatGoogleGenerativeAI(
        model=get_settings().llm_model or "gemini-2.5-flash",
        temperature=0,
        max_output_tokens=max_tokens,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT_SECONDS,
        thinking_budget=0,  # same reason: keep the whole token budget for the answer
    )
    return claude.with_fallbacks([gemini])


def get_embeddings() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )

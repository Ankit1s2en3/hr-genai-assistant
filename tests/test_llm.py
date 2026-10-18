"""Checks how the chat model is configured. Nothing here calls an API."""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from app.llm import get_chat_model, model_name


def test_chat_model_is_claude_with_gemini_fallback():
    model = get_chat_model("claude-haiku-5-5", max_tokens=256)

    claude = model.runnable
    assert isinstance(claude, ChatAnthropic)
    assert claude.model == "claude-haiku-5-5"
    assert claude.max_tokens == 256
    assert claude.max_retries == 2
    # Haiku 5.5 rejects a non-default temperature; sending one would make every
    # Claude call fail and silently fall back to Gemini.
    assert "temperature" not in claude._get_request_payload([("human", "hi")])

    (gemini,) = model.fallbacks
    assert isinstance(gemini, ChatGoogleGenerativeAI)
    assert gemini.temperature == 0
    assert gemini.max_output_tokens == 256


def test_falls_back_to_gemini_and_logs_the_claude_error(monkeypatch, caplog):
    def claude_fails(self, *args, **kwargs):
        raise RuntimeError("Claude is down")

    def gemini_answers(self, *args, **kwargs):
        message = AIMessage(content="Hi from Gemini", response_metadata={"model_name": "gemini-2.5-flash"})
        return ChatResult(generations=[ChatGeneration(message=message)])

    # Replace the network calls inside each model; everything above them runs for real.
    monkeypatch.setattr(ChatAnthropic, "_generate", claude_fails)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_generate", gemini_answers)

    with caplog.at_level(logging.WARNING, logger="app.llm"):
        reply = get_chat_model("claude-haiku-5-5", max_tokens=800).invoke("hello")

    assert reply.text == "Hi from Gemini"
    assert model_name(reply) == "gemini-2.5-flash"
    assert "falling back to Gemini" in caplog.text
    assert "Claude is down" in caplog.text

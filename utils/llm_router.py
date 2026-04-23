"""
utils/llm_router.py
Unified LLM call interface.
Supports: gpt-4o (OpenAI), claude (Anthropic), groq (FREE cloud LLaMA-3), llama-3 (Ollama local).
"""
from __future__ import annotations
import json
from loguru import logger
from config.settings import settings


async def call_llm(
    prompt: str,
    system: str = "You are a financial analysis assistant.",
    model_key: str = "groq",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """
    Route an LLM call to the appropriate provider.

    Args:
        prompt:     User message content.
        system:     System prompt.
        model_key:  "groq" | "gpt-4o" | "claude" | "llama-3"
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        json_mode:  If True, instruct model to return valid JSON only.

    Returns:
        Response text from the model.
    """
    if json_mode:
        system += "\nRespond ONLY with valid JSON. No markdown, no explanation."

    logger.debug(f"LLM call → model={model_key}")

    if model_key in ("gpt-4o", "openai"):
        return await _call_openai(prompt, system, temperature, max_tokens, json_mode)
    elif model_key in ("claude", "anthropic"):
        return await _call_anthropic(prompt, system, temperature, max_tokens)
    elif model_key in ("groq",):
        return await _call_groq(prompt, system, temperature, max_tokens, json_mode)
    elif model_key in ("llama-3", "ollama"):
        return await _call_ollama(prompt, system, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown model_key: {model_key!r}")


# ─── OpenAI ──────────────────────────────────────────────────────────────────

async def _call_openai(
    prompt: str, system: str, temperature: float, max_tokens: int, json_mode: bool
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    kwargs: dict = dict(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ─── Anthropic Claude ─────────────────────────────────────────────────────────

async def _call_anthropic(
    prompt: str, system: str, temperature: float, max_tokens: int
) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    resp = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.content[0].text


# ─── Groq (FREE cloud LLaMA-3 — no Ollama needed) ────────────────────────────

async def _call_groq(
    prompt: str, system: str, temperature: float, max_tokens: int, json_mode: bool
) -> str:
    """
    Call LLaMA-3 via Groq's FREE cloud API.
    Uses OpenAI-compatible format — no extra dependencies needed.

    Get your free API key at: https://console.groq.com/keys
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    kwargs: dict = dict(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ─── Ollama (local LLaMA-3) ───────────────────────────────────────────────────

async def _call_ollama(
    prompt: str, system: str, temperature: float, max_tokens: int
) -> str:
    import httpx

    payload = {
        "model": "llama3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

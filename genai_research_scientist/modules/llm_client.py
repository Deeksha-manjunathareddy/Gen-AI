from __future__ import annotations

"""
Lightweight wrapper around LiteLLM so the app can talk to multiple
LLM providers (OpenAI, Gemini, Claude, Groq, etc.) with a single API.

API keys are read from environment variables as expected by LiteLLM
and can be conveniently stored in a local `.env` file, for example:

    # For Anthropic/Claude:
    PLANNING_MODEL=anthropic/claude-3-5-sonnet-20240620
    ANTHROPIC_API_KEY=sk-ant-...

    # For Google/Gemini:
    PLANNING_MODEL=gemini/gemini-1.5-pro
    GOOGLE_API_KEY=... (or GEMINI_API_KEY=...)

    # For OpenAI:
    PLANNING_MODEL=gpt-4o
    OPENAI_API_KEY=sk-...

    # For Groq:
    PLANNING_MODEL=groq/llama3-70b-8192
    GROQ_API_KEY=...

See MODEL_CONFIGURATION.md for detailed examples and supported models.
"""

import os
import json
from typing import Optional, TypeVar, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # allow local .env usage

# Get PLANNING_MODEL from environment
PLANNING_MODEL = os.getenv("PLANNING_MODEL")

try:
    from litellm import completion, acompletion
except Exception:  # pragma: no cover - litellm is optional at runtime
    completion = None  # type: ignore
    acompletion = None  # type: ignore

T = TypeVar("T", bound=BaseModel)


def get_api_key_for_model(model: str) -> str:
    """
    Get the appropriate API key based on the model provider.
    LiteLLM reads from environment variables, but we need to ensure they're set.
    """
    model_lower = model.lower()
    
    if "anthropic" in model_lower or "claude" in model_lower:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set for Anthropic model")
        return api_key
    elif "openai" in model_lower or "gpt" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI model")
        return api_key
    elif "gemini" in model_lower or "google" in model_lower:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set for Google model")
        return api_key
    else:
        # For other providers, try to get from environment
        # LiteLLM will handle the specific key name
        return os.getenv("ANTHROPIC_API_KEY", "")


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Call an LLM via LiteLLM and return the response text (synchronous, for backward compatibility).

    If LiteLLM is not installed or an error occurs, return None so callers
    can gracefully fall back to rule-based/template logic.
    """
    if completion is None or not model:
        return None

    try:
        resp = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        # LiteLLM normalizes to .message for chat-style models
        msg = getattr(choice, "message", getattr(choice, "delta", None))
        if isinstance(msg, dict):
            return msg.get("content")
        if hasattr(msg, "content"):
            return msg.content
        return None
    except Exception:
        # Intentionally swallow errors and let caller fall back.
        return None


async def call_llm_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: Type[T],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Optional[T]:
    """
    Call an LLM via LiteLLM with async and return a Pydantic structured output.

    Args:
        model: LiteLLM model string (e.g., "anthropic/claude-3-5-sonnet-20240620")
        system_prompt: System prompt for the LLM
        user_prompt: User prompt for the LLM
        response_format: Pydantic BaseModel class for structured output
        temperature: Temperature for generation (default 0.0 for structured outputs)
        max_tokens: Maximum tokens (optional)

    Returns:
        Validated Pydantic model instance or None if error occurs
    """
    if acompletion is None or not model:
        return None

    # Ensure API key is set for the model
    try:
        get_api_key_for_model(model)
    except ValueError:
        # If API key not set, LiteLLM will handle it or fail gracefully
        pass

    try:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,  # Pydantic class passed directly
        )

        # Extract content from LiteLLM response
        content = resp.choices[0].message.content

        # Parse the JSON content into Pydantic model
        if isinstance(content, str):
            data = json.loads(content)
            return response_format.model_validate(data)
        elif isinstance(content, dict):
            return response_format.model_validate(content)
        elif isinstance(content, response_format):
            return content
        else:
            return None
    except Exception:
        # Intentionally swallow errors and let caller fall back.
        return None



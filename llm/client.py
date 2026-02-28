"""
Unified LLM client wrapper.

Supports multiple providers through a single interface:
  - deepseek  (default): uses OpenAI-compatible API at api.deepseek.com
  - anthropic           : uses Anthropic's native SDK
  - openai_compatible   : any OpenAI-compatible endpoint (set base_url)

Usage:
    from llm.client import LLMClient, LLMConfig

    # DeepSeek (default)
    client = LLMClient(LLMConfig(api_key="sk-..."))

    # Anthropic
    client = LLMClient(LLMConfig(provider="anthropic", api_key="sk-ant-..."))

    # Custom OpenAI-compatible endpoint
    client = LLMClient(LLMConfig(
        provider="openai_compatible",
        api_key="sk-...",
        base_url="https://your-endpoint/v1",
    ))

    response = client.chat(
        model="deepseek-chat",
        system="You are a helpful assistant.",
        user="Hello!",
        max_tokens=1024,
    )
"""

import json
from dataclasses import dataclass, field
from typing import Optional


# Default models per provider
_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "deepseek": {
        "main": "deepseek-chat",
        "summarizer": "deepseek-chat",
    },
    "anthropic": {
        "main": "claude-sonnet-4-6",
        "summarizer": "claude-haiku-4-5-20251001",
    },
    "openai_compatible": {
        "main": "gpt-4o",
        "summarizer": "gpt-4o-mini",
    },
}

# Hard cap on max_tokens per provider.
# None means "no client-side cap" (rely on API to enforce).
_MAX_OUTPUT_TOKENS: dict[str, Optional[int]] = {
    "deepseek": 8192,
    "anthropic": None,
    "openai_compatible": None,
}

# Base URLs for known providers
_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "openai_compatible": "",  # must be supplied by caller
}


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: str = "deepseek"       # "deepseek" | "anthropic" | "openai_compatible"
    api_key: str = ""
    base_url: Optional[str] = None   # Override the default base URL
    max_output_tokens: Optional[int] = None  # Override per-provider token cap (None = use default)


class LLMClient:
    """
    Thin wrapper around LLM provider SDKs.

    All callers use ``client.chat(model, system, user, max_tokens) -> str``
    regardless of the underlying provider.
    """

    def __init__(self, config: LLMConfig):
        self.provider = config.provider
        self._config = config

        # Resolve effective token cap: explicit config > provider default > None
        if config.max_output_tokens is not None:
            self._max_tokens_cap: Optional[int] = config.max_output_tokens
        else:
            self._max_tokens_cap = _MAX_OUTPUT_TOKENS.get(config.provider)

        if config.provider == "anthropic":
            from anthropic import Anthropic
            self._backend = Anthropic(api_key=config.api_key)
        else:
            # DeepSeek and any other OpenAI-compatible provider
            from openai import OpenAI
            base_url = config.base_url or _BASE_URLS.get(config.provider, "")
            if not base_url:
                raise ValueError(
                    f"provider='{config.provider}' requires base_url to be set."
                )
            self._backend = OpenAI(api_key=config.api_key, base_url=base_url)

    def _cap_tokens(self, max_tokens: int) -> int:
        """Clamp max_tokens to the provider's hard limit (if any)."""
        if self._max_tokens_cap is not None:
            return min(max_tokens, self._max_tokens_cap)
        return max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a single-turn chat request and return the assistant response text.

        Args:
            model:      Model identifier (provider-specific).
            system:     System prompt.
            user:       User message.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's reply as a plain string.
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(model, system, user, max_tokens)
        else:
            return self._chat_openai_compatible(model, system, user, max_tokens)

    # ------------------------------------------------------------------
    # Helper: default model names for this provider
    # ------------------------------------------------------------------

    def chat_structured(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        schema_name: str,
        json_schema: dict,
    ) -> dict:
        """
        Send a single-turn chat request and return the assistant response as a
        parsed dict, using provider-native structured output.

        For Anthropic: uses the tools API with tool_choice forced to schema_name,
        guaranteeing valid JSON that conforms to json_schema.
        For OpenAI-compatible: uses response_format={"type": "json_object"} and
        json.loads() on the result (schema is enforced by the prompt, not the API).

        Args:
            model:       Model identifier (provider-specific).
            system:      System prompt.
            user:        User message.
            max_tokens:  Maximum tokens in the response.
            schema_name: Tool name used for Anthropic tool_choice forcing.
            json_schema: JSON Schema dict describing the expected output shape.

        Returns:
            Parsed dict. Returns {} on any extraction or parse failure.
        """
        if self.provider == "anthropic":
            return self._chat_structured_anthropic(
                model, system, user, max_tokens, schema_name, json_schema
            )
        else:
            return self._chat_structured_openai_compatible(
                model, system, user, max_tokens, schema_name, json_schema
            )

    # ------------------------------------------------------------------
    # Helper: default model names for this provider
    # ------------------------------------------------------------------

    def default_model(self, role: str = "main") -> str:
        """
        Return the default model name for this provider.

        Args:
            role: "main" (planning/implementation) or "summarizer" (batch jobs).
        """
        provider_defaults = _DEFAULT_MODELS.get(self.provider, _DEFAULT_MODELS["deepseek"])
        return provider_defaults.get(role, provider_defaults["main"])

    # ------------------------------------------------------------------
    # Private: provider-specific implementations
    # ------------------------------------------------------------------

    def _chat_anthropic(self, model: str, system: str, user: str, max_tokens: int) -> str:
        response = self._backend.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _chat_openai_compatible(self, model: str, system: str, user: str, max_tokens: int) -> str:
        response = self._backend.chat.completions.create(
            model=model,
            max_tokens=self._cap_tokens(max_tokens),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def _chat_structured_anthropic(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        schema_name: str,
        json_schema: dict,
    ) -> dict:
        """Anthropic tools-API structured output — block.input is already a dict."""
        tool_def = {
            "name": schema_name,
            "description": f"Structured output: {schema_name}",
            "input_schema": json_schema,
        }
        response = self._backend.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": schema_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # already a Python dict
        return {}

    def _chat_structured_openai_compatible(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        schema_name: str,
        json_schema: dict,
    ) -> dict:
        """OpenAI-compatible structured output via json_object response_format."""
        response = self._backend.chat.completions.create(
            model=model,
            max_tokens=self._cap_tokens(max_tokens),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}


# ------------------------------------------------------------------
# Factory helper
# ------------------------------------------------------------------

def create_client_from_env(provider: str = "deepseek", base_url: Optional[str] = None) -> LLMClient:
    """
    Create an LLMClient using API key from the environment.

    Environment variables checked (in order):
      - DEEPSEEK_API_KEY   (when provider == "deepseek")
      - ANTHROPIC_API_KEY  (when provider == "anthropic")
      - LLM_API_KEY        (fallback for any provider)

    Args:
        provider: Provider name ("deepseek", "anthropic", "openai_compatible").
        base_url: Override base URL (required for "openai_compatible").

    Returns:
        A configured LLMClient.

    Raises:
        SystemExit if no API key is found.
    """
    import os
    import sys

    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    primary_env = env_map.get(provider, "LLM_API_KEY")

    api_key = (
        os.environ.get(primary_env)
        or os.environ.get("LLM_API_KEY")
    )

    if not api_key:
        sys.exit(
            f"Error: API key not found. "
            f"Set the {primary_env} (or LLM_API_KEY) environment variable."
        )

    return LLMClient(LLMConfig(provider=provider, api_key=api_key, base_url=base_url))

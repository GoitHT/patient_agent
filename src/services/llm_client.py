from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx

from utils import parse_json_with_retry


class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: Callable[[], dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> tuple[dict[str, Any], bool, str]:
        """Return (json_obj, used_fallback, raw_text)."""
    
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate plain text response."""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_s: float = 120.0  # 增加到120秒以支持复杂诊断推理


class DeepSeekLLMClient:
    """DeepSeek Chat Completions (OpenAI-compatible) client.

    Security:
    - Do NOT hardcode API keys in code. Use env var `DEEPSEEK_API_KEY`.
    - Avoid sending sensitive patient identifiers to any external API.
    """

    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config

    @staticmethod
    def from_env() -> "DeepSeekLLMClient":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing env var: DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        timeout_s = float(os.getenv("DEEPSEEK_TIMEOUT_S", "120"))  # 默认120秒
        return DeepSeekLLMClient(
            DeepSeekConfig(api_key=api_key, base_url=base_url, model=model, timeout_s=timeout_s)
        )

    def _chat(self, *, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, json_mode: bool = True) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # Add JSON mode only when requested
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected DeepSeek response shape: {data}") from e

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: Callable[[], dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> tuple[dict[str, Any], bool, str]:
        raw = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        obj, used_fallback = parse_json_with_retry(raw, fallback=fallback)
        return obj, used_fallback, raw
    
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate plain text response (not JSON)."""
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )


def build_llm_client(mode: str | None) -> LLMClient | None:
    """Factory used by CLI/router. `mode` can be: None/'mock'/'deepseek'."""

    if mode is None or mode == "mock":
        return None
    if mode == "deepseek":
        return DeepSeekLLMClient.from_env()
    raise ValueError(f"Unknown LLM mode: {mode}")


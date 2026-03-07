from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx

from utils import parse_json_with_retry, get_logger

logger = get_logger(__name__)

# 禁用httpx的详细日志输出（避免显示每次HTTP请求）
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    base_url: str
    model: str
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_delay: float = 2.0


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
        # 从环境变量读取所有配置（无默认值，确保配置来自.env文件）
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "缺少环境变量: DEEPSEEK_API_KEY\n"
                "请在 .env 文件中设置: DEEPSEEK_API_KEY=your-api-key"
            )
        
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError(
                "缺少环境变量: DEEPSEEK_BASE_URL\n"
                "请在 .env 文件中设置: DEEPSEEK_BASE_URL=https://api.apiyi.com/v1"
            )
        
        model = os.getenv("DEEPSEEK_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "缺少环境变量: DEEPSEEK_MODEL\n"
                "请在 .env 文件中设置: DEEPSEEK_MODEL=gpt-4o-mini"
            )
        
        timeout_s = float(os.getenv("DEEPSEEK_TIMEOUT_S", "120"))
        max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
        retry_delay = float(os.getenv("DEEPSEEK_RETRY_DELAY", "2.0"))
        
        return DeepSeekLLMClient(
            DeepSeekConfig(
                api_key=api_key, 
                base_url=base_url, 
                model=model, 
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
        )

    @staticmethod
    def from_env_chatgpt() -> "DeepSeekLLMClient":
        """使用 .env 中 CHATGPT_* 环境变量构建客户端（OpenAI 兼容接口）。"""
        api_key = os.getenv("CHATGPT_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "缺少环境变量: CHATGPT_API_KEY\n"
                "请在 .env 文件中设置: CHATGPT_API_KEY=your-api-key"
            )

        base_url = os.getenv("CHATGPT_API_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError(
                "缺少环境变量: CHATGPT_API_BASE_URL\n"
                "请在 .env 文件中设置: CHATGPT_API_BASE_URL=https://api.apiyi.com/v1"
            )

        model = os.getenv("CHATGPT_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "缺少环境变量: CHATGPT_MODEL\n"
                "请在 .env 文件中设置: CHATGPT_MODEL=gpt-4o"
            )

        timeout_s = float(os.getenv("CHATGPT_TIMEOUT_S", "120"))
        max_retries = int(os.getenv("CHATGPT_MAX_RETRIES", "3"))
        retry_delay = float(os.getenv("CHATGPT_RETRY_DELAY", "2.0"))

        return DeepSeekLLMClient(
            DeepSeekConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
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
            "stream": False,  # 明确禁用流式输出
        }
        # Add JSON mode only when requested (某些API可能不支持)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        # 调试日志
        logger.debug(f"📡 API请求: {url}")
        logger.debug(f"🔑 API Key (前8位): {self.config.api_key[:8]}...")
        logger.debug(f"📋 模型: {self.config.model}")
        
        # 重试机制
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                # 配置httpx客户端，增加连接池和超时设置
                with httpx.Client(
                    timeout=httpx.Timeout(
                        connect=10.0,  # 连接超时
                        read=self.config.timeout_s,  # 读取超时
                        write=10.0,  # 写入超时
                        pool=5.0  # 连接池超时
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10
                    )
                ) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                
                # 成功获取响应
                try:
                    return str(data["choices"][0]["message"]["content"])
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Unexpected DeepSeek response shape: {data}") from e
                    
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError) as e:
                last_exception = e
                
                # 判断是否需要重试
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (attempt + 1)  # 指数退避
                    logger.warning(
                        f"⚠️  DeepSeek API调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e.__class__.__name__}: {str(e)[:100]}"
                    )
                    logger.info(f"   等待 {wait_time:.1f}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ DeepSeek API调用失败，已达最大重试次数 ({self.config.max_retries})")
                    
            except httpx.HTTPStatusError as e:
                # HTTP状态错误（如429, 500等）
                if e.response.status_code == 401:
                    # 401错误：认证失败，提供详细信息
                    try:
                        error_detail = e.response.json()
                        logger.error(f"❌ API认证失败 (401)")
                        logger.error(f"   URL: {url}")
                        logger.error(f"   API Key (前8位): {self.config.api_key[:8]}...")
                        logger.error(f"   模型: {self.config.model}")
                        logger.error(f"   错误详情: {error_detail}")
                    except Exception:
                        logger.error(f"❌ API认证失败 (401): {e.response.text[:200]}")
                    
                    raise RuntimeError(
                        f"API认证失败 (401)。请检查:\n"
                        f"1. API密钥是否正确: {self.config.api_key[:12]}...\n"
                        f"2. API URL是否正确: {self.config.base_url}\n"
                        f"3. 密钥是否有权限访问模型: {self.config.model}"
                    ) from e
                else:
                    logger.error(f"❌ API返回错误状态: {e.response.status_code}")
                    logger.error(f"   响应内容: {e.response.text[:200]}")
                    raise
                
            except Exception as e:
                # 其他未预期的错误
                logger.error(f"❌ DeepSeek API调用出现未知错误: {e.__class__.__name__}: {str(e)[:100]}")
                raise
        
        # 所有重试都失败
        if last_exception:
            raise RuntimeError(
                f"DeepSeek API调用失败（已重试{self.config.max_retries}次）: {last_exception.__class__.__name__}: {str(last_exception)}"
            ) from last_exception
        else:
            raise RuntimeError("DeepSeek API调用失败（原因未知）")

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


def build_llm_client(mode: str | None) -> LLMClient:
    """Factory used by CLI/router. `mode` can be: 'deepseek'/'chatgpt'."""

    if mode == "deepseek":
        return DeepSeekLLMClient.from_env()
    if mode == "chatgpt":
        return DeepSeekLLMClient.from_env_chatgpt()
    raise ValueError(f"Unknown LLM mode: {mode!r}，可选值: deepseek / chatgpt")


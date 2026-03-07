from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ChatGenerationConfig:
    model: str
    temperature: float | None = 0.6
    top_p: float | None = 0.95
    max_tokens: int | None = 1536
    stop: list[str] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleLLM:
    """OpenAI-compatible chat client for OpenAI/vLLM-compatible endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    def chat(self, messages: list[dict[str, Any]], gen_cfg: ChatGenerationConfig) -> str:
        payload: dict[str, Any] = {
            "model": gen_cfg.model,
            "messages": messages,
        }
        if gen_cfg.temperature is not None:
            payload["temperature"] = gen_cfg.temperature
        if gen_cfg.top_p is not None:
            payload["top_p"] = gen_cfg.top_p
        if gen_cfg.max_tokens is not None:
            payload["max_tokens"] = gen_cfg.max_tokens
        if gen_cfg.stop:
            payload["stop"] = gen_cfg.stop
        if gen_cfg.extra_body:
            payload["extra_body"] = gen_cfg.extra_body

        last_err: Exception | None = None
        for i in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**payload)
                content = resp.choices[0].message.content
                return _content_to_text(content)
            except Exception as err:  # pragma: no cover
                last_err = err
                if i < self.max_retries - 1:
                    time.sleep(self.retry_backoff_sec * (2**i))
                    continue
                break
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")


def build_llm_backend(llm_cfg: dict[str, Any]) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url=str(llm_cfg.get("base_url") or "https://api.openai.com/v1"),
        api_key=str(llm_cfg.get("api_key") or "EMPTY"),
        timeout=float(llm_cfg.get("timeout", 120.0)),
        max_retries=int(llm_cfg.get("max_retries", 3)),
        retry_backoff_sec=float(llm_cfg.get("retry_backoff_sec", 1.0)),
    )


def resolve_agent_llm_config(
    global_llm_cfg: dict[str, Any], agent_cfg: dict[str, Any], agent_key: str
) -> dict[str, Any]:
    merged_cfg = dict(global_llm_cfg)
    agent_llm_cfg = agent_cfg.get("llm")
    if agent_llm_cfg is None:
        return merged_cfg
    if not isinstance(agent_llm_cfg, dict):
        raise ValueError(f"`agents.{agent_key}.llm` must be a dict")
    merged_cfg.update(agent_llm_cfg)
    return merged_cfg


def build_generation_config(llm_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> ChatGenerationConfig:
    model = agent_cfg.get("model") or llm_cfg.get("model")
    if not model:
        raise ValueError(
            "Model is required in `llm.model`, `agents.<agent>.llm.model`, "
            "or `agents.<agent>.model`"
        )
    return ChatGenerationConfig(
        model=model,
        temperature=agent_cfg.get("temperature", llm_cfg.get("temperature", 0.6)),
        top_p=agent_cfg.get("top_p", llm_cfg.get("top_p", 0.95)),
        max_tokens=agent_cfg.get("max_tokens", llm_cfg.get("max_tokens", 1536)),
        stop=agent_cfg.get("stop", llm_cfg.get("stop")),
        extra_body=agent_cfg.get("extra_body", llm_cfg.get("extra_body", {})),
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)

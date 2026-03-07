from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand_env_in_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, "")

    return ENV_PATTERN.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _expand_env_in_string(value)
    return value


def _get_non_empty_env(*names: str) -> tuple[str, str] | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return name, value
    return None


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = config
    for key in path[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            joined = ".".join(path[:-1])
            raise ValueError(f"Cannot set `{'.'.join(path)}` because `{joined}` is not a dict")
        current = child
    current[path[-1]] = value


def _cast_env_value(env_name: str, raw: str, caster: Callable[[str], Any]) -> Any:
    try:
        return caster(raw)
    except Exception as err:  # pragma: no cover
        raise ValueError(f"Invalid value for env `{env_name}`: {raw}") from err


def _apply_single_env_override(
    config: dict[str, Any],
    env_names: tuple[str, ...],
    path: tuple[str, ...],
    caster: Callable[[str], Any] = str,
) -> None:
    env_value = _get_non_empty_env(*env_names)
    if env_value is None:
        return
    env_name, raw = env_value
    _set_nested(config, path, _cast_env_value(env_name, raw, caster))


def _apply_llm_env_overrides(config: dict[str, Any]) -> None:
    common_overrides: tuple[tuple[tuple[str, ...], tuple[str, ...], Callable[[str], Any]], ...] = (
        (("SEARCH_MAS_LLM_BASE_URL", "OPENAI_BASE_URL"), ("llm", "base_url"), str),
        (("SEARCH_MAS_LLM_API_KEY", "OPENAI_API_KEY"), ("llm", "api_key"), str),
        (("SEARCH_MAS_LLM_MODEL", "OPENAI_MODEL"), ("llm", "model"), str),
        (("SEARCH_MAS_LLM_TIMEOUT",), ("llm", "timeout"), float),
        (("SEARCH_MAS_LLM_MAX_RETRIES",), ("llm", "max_retries"), int),
        (("SEARCH_MAS_LLM_RETRY_BACKOFF_SEC",), ("llm", "retry_backoff_sec"), float),
        (("SEARCH_MAS_LLM_TEMPERATURE",), ("llm", "temperature"), float),
        (("SEARCH_MAS_LLM_TOP_P",), ("llm", "top_p"), float),
        (("SEARCH_MAS_LLM_MAX_TOKENS",), ("llm", "max_tokens"), int),
    )
    for env_names, path, caster in common_overrides:
        _apply_single_env_override(config, env_names, path, caster)

    for agent_name in ("verifier", "searcher", "answerer"):
        upper = agent_name.upper()
        agent_overrides: tuple[tuple[tuple[str, ...], tuple[str, ...], Callable[[str], Any]], ...] = (
            ((f"SEARCH_MAS_{upper}_LLM_BASE_URL",), ("agents", agent_name, "llm", "base_url"), str),
            ((f"SEARCH_MAS_{upper}_LLM_API_KEY",), ("agents", agent_name, "llm", "api_key"), str),
            ((f"SEARCH_MAS_{upper}_LLM_MODEL",), ("agents", agent_name, "llm", "model"), str),
            ((f"SEARCH_MAS_{upper}_LLM_TIMEOUT",), ("agents", agent_name, "llm", "timeout"), float),
            (
                (f"SEARCH_MAS_{upper}_LLM_MAX_RETRIES",),
                ("agents", agent_name, "llm", "max_retries"),
                int,
            ),
            (
                (f"SEARCH_MAS_{upper}_LLM_RETRY_BACKOFF_SEC",),
                ("agents", agent_name, "llm", "retry_backoff_sec"),
                float,
            ),
            ((f"SEARCH_MAS_{upper}_TEMPERATURE",), ("agents", agent_name, "temperature"), float),
            ((f"SEARCH_MAS_{upper}_TOP_P",), ("agents", agent_name, "top_p"), float),
            ((f"SEARCH_MAS_{upper}_MAX_TOKENS",), ("agents", agent_name, "max_tokens"), int),
        )
        for env_names, path, caster in agent_overrides:
            _apply_single_env_override(config, env_names, path, caster)


def _apply_search_env_overrides(config: dict[str, Any]) -> None:
    search_overrides: tuple[tuple[tuple[str, ...], tuple[str, ...], Callable[[str], Any]], ...] = (
        (("SEARCH_MAS_RETRIEVAL_SERVICE_URL",), ("search", "retrieval_service_url"), str),
        (("SEARCH_MAS_SEARCH_TOPK",), ("search", "topk"), int),
        (("SEARCH_MAS_SEARCH_TIMEOUT",), ("search", "timeout"), float),
    )
    for env_names, path, caster in search_overrides:
        _apply_single_env_override(config, env_names, path, caster)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load yaml config with `${ENV_VAR}` / `$ENV_VAR` expansion and env overrides."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict in file: {path}")
    expanded = _expand_env(config)
    if not isinstance(expanded, dict):
        raise ValueError(f"Config must be a dict after env expansion: {path}")
    _apply_llm_env_overrides(expanded)
    _apply_search_env_overrides(expanded)
    return expanded

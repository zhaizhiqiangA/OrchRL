from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _config_get(config: Any, key: str):
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(key)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(config, key, None)



def _legacy_model_name(model_path: str) -> str:
    return model_path
    # return "/".join(str(model_path).split("/")[-2:])



def resolve_served_model_name(rollout_config: Any, model_path: str) -> str:
    explicit_name = _config_get(rollout_config, "served_model_name")
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name
    return _legacy_model_name(model_path)



def resolve_policy_server_name(policy_name: str, ppo_config: Any) -> str:
    if ppo_config is None:
        return policy_name
    actor_rollout_ref = _config_get(ppo_config, "actor_rollout_ref")
    rollout_config = _config_get(actor_rollout_ref, "rollout")
    model_config = _config_get(actor_rollout_ref, "model")
    model_path = _config_get(model_config, "path")

    explicit_name = _config_get(rollout_config, "served_model_name")
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name
    if model_path is None:
        return policy_name
    if "checkpoint" in str(model_path):
        return str(model_path)
    return _legacy_model_name(str(model_path))

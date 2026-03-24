from __future__ import annotations

from typing import Any


def build_drift_artifact(
    *,
    messages: list[dict[str, Any]],
    runtime_prompt_ids: list[int] | None,
    rerendered_prompt_ids: list[int] | None,
    response_ids: list[int] | None,
    response_logprobs: list[float] | None,
    render_fingerprint: dict[str, Any] | None,
    sampling_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "messages": messages,
        "runtime_prompt_ids": runtime_prompt_ids,
        "rerendered_prompt_ids": rerendered_prompt_ids,
        "response_ids": response_ids,
        "response_logprobs": response_logprobs,
        "render_fingerprint": render_fingerprint or {},
        "sampling_fingerprint": sampling_fingerprint or {},
        "mismatch": runtime_prompt_ids != rerendered_prompt_ids,
    }

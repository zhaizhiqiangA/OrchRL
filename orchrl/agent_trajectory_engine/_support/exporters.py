from __future__ import annotations

from typing import Any

from ..datatypes import TurnData


def export_tokenized_turn(turn: TurnData) -> dict[str, Any]:
    if turn.prompt_ids is None:
        raise ValueError("recorded prompt_ids required for token-truth export")
    if turn.token_ids is None:
        raise ValueError("response token_ids required for token-truth export")

    return {
        "agent_role": turn.agent_role,
        "turn_index": turn.turn_index,
        "prompt_ids": list(turn.prompt_ids),
        "response_ids": list(turn.token_ids),
        "response_logprobs": list(turn.logprobs) if turn.logprobs is not None else None,
        "replayed": turn.replayed,
        "branch_phase": turn.branch_phase,
        "routed_experts": turn.routed_experts,
    }

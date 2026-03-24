from __future__ import annotations

import pytest

from trajectory import BranchResult, EpisodeResult, EpisodeTrajectory, TreeEpisodeResult, TurnData

from orchrl.trainer.mate_dataproto_adapter import tree_episodes_to_policy_batches


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        return [len(messages), 100 + len(messages)]


_DEFAULT_PROMPT_IDS = object()


def _turn(
    role: str,
    turn_index: int,
    timestamp: float,
    *,
    replayed: bool = False,
    prompt_ids: list[int] | None | object = _DEFAULT_PROMPT_IDS,
) -> TurnData:
    metadata = {}
    if replayed:
        metadata["replayed"] = True
    resolved_prompt_ids = [1000 + turn_index, 2000 + turn_index] if prompt_ids is _DEFAULT_PROMPT_IDS else prompt_ids
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[turn_index + 1, turn_index + 2],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        timestamp=timestamp,
        prompt_ids=resolved_prompt_ids,
        metadata=metadata,
    )


def _episode(episode_id: str, trajectories: dict[str, list[TurnData]], *, sample_idx: int = 0) -> EpisodeResult:
    return EpisodeResult(
        trajectory=EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories=trajectories,
            metadata={},
        ),
        rewards={"verifier": 1.0, "searcher": 2.0},
        final_reward=1.0,
        metadata={"prompt_group_id": "prompt-7", "sample_idx": sample_idx},
    )


def test_tree_episodes_to_policy_batches_uses_recorded_prompt_ids() -> None:
    tree_episode = TreeEpisodeResult(
        pilot_result=_episode(
            "pilot",
            {
                "verifier": [_turn("verifier", 0, 1.0, prompt_ids=[11, 22, 33])],
                "searcher": [_turn("searcher", 0, 2.0, prompt_ids=[44, 55])],
            },
        ),
        branch_results=[],
        prompt="prompt",
        tree_metadata={},
    )

    batches = tree_episodes_to_policy_batches(
        episodes=[tree_episode],
        tokenizer_dict={"policy_v": _Tokenizer(), "policy_s": _Tokenizer()},
        role_policy_mapping={"verifier": "policy_v", "searcher": "policy_s"},
        role_index_mapping={"verifier": 0, "searcher": 1},
        max_prompt_length=32,
        max_response_length=32,
        backend_mode="canonical",
    )

    assert batches["policy_v"].batch["prompts"][0].tolist()[-3:] == [11, 22, 33]
    assert batches["policy_s"].batch["prompts"][0].tolist()[-2:] == [44, 55]


def test_tree_episodes_to_policy_batches_fails_when_prompt_ids_missing() -> None:
    tree_episode = TreeEpisodeResult(
        pilot_result=_episode(
            "pilot",
            {
                "verifier": [_turn("verifier", 0, 1.0, prompt_ids=None)],
            },
        ),
        branch_results=[],
        prompt="prompt",
        tree_metadata={},
    )

    with pytest.raises(ValueError, match="prompt_ids"):
        tree_episodes_to_policy_batches(
            episodes=[tree_episode],
            tokenizer_dict={"policy_v": _Tokenizer()},
            role_policy_mapping={"verifier": "policy_v"},
            role_index_mapping={"verifier": 0},
            max_prompt_length=32,
            max_response_length=32,
            backend_mode="canonical",
        )


def test_tree_episodes_to_policy_batches_emits_branch_aware_uids() -> None:
    pilot = _episode(
        "pilot",
        {
            "verifier": [_turn("verifier", 0, 1.0)],
            "searcher": [_turn("searcher", 0, 2.0)],
        },
    )
    branch_episode = _episode(
        "branch",
        {
            "verifier": [_turn("verifier", 0, 3.0, replayed=True), _turn("verifier", 1, 5.0)],
            "searcher": [_turn("searcher", 0, 4.0)],
        },
        sample_idx=1,
    )
    tree_episode = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(
                episode_result=branch_episode,
                branch_turn=1,
                branch_agent_role="searcher",
                parent_episode_id=pilot.trajectory.episode_id,
            )
        ],
        prompt="prompt",
        tree_metadata={},
    )

    batches = tree_episodes_to_policy_batches(
        episodes=[tree_episode],
        tokenizer_dict={"policy_v": _Tokenizer(), "policy_s": _Tokenizer()},
        role_policy_mapping={"verifier": "policy_v", "searcher": "policy_s"},
        role_index_mapping={"verifier": 0, "searcher": 1},
        max_prompt_length=32,
        max_response_length=32,
        backend_mode="canonical",
    )

    assert batches["policy_s"].non_tensor_batch["uid"][-1] == "prompt-7:1:b1"
    assert batches["policy_v"].non_tensor_batch["uid"][-1] == "prompt-7:0:b1:t2"

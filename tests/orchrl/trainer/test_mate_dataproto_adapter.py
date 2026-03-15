from __future__ import annotations

from trajectory import BranchResult, EpisodeResult, EpisodeTrajectory, TreeEpisodeResult, TurnData

from orchrl.trainer.mate_dataproto_adapter import tree_episodes_to_policy_batches


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        return [len(messages), 100 + len(messages)]


def _turn(role: str, turn_index: int, timestamp: float, *, replayed: bool = False) -> TurnData:
    metadata = {}
    if replayed:
        metadata["replayed"] = True
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[turn_index + 1, turn_index + 2],
        logprobs=[-0.1, -0.2],
        finish_reason="stop",
        timestamp=timestamp,
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


def test_tree_episodes_to_policy_batches_keeps_pilot_uid_compatible() -> None:
    tree_episode = TreeEpisodeResult(
        pilot_result=_episode(
            "pilot",
            {
                "verifier": [_turn("verifier", 0, 1.0)],
                "searcher": [_turn("searcher", 0, 2.0)],
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
    )

    assert batches["policy_v"].non_tensor_batch["uid"] == ["prompt-7:0"]
    assert batches["policy_s"].non_tensor_batch["uid"] == ["prompt-7:1"]


def test_tree_episodes_to_policy_batches_skips_replayed_prefix_turns() -> None:
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
            "verifier": [_turn("verifier", 0, 3.0, replayed=True)],
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
    )

    assert list(batches["policy_v"].non_tensor_batch["episode_id"]) == ["pilot"]
    assert list(batches["policy_s"].non_tensor_batch["episode_id"]) == ["pilot", "branch"]


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
    )

    assert batches["policy_s"].non_tensor_batch["uid"][-1] == "prompt-7:1:b1"
    assert batches["policy_v"].non_tensor_batch["uid"][-1] == "prompt-7:0:b1:t2"

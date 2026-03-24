from __future__ import annotations

from dataclasses import dataclass

import pytest

from orchrl.agent_trajectory_engine import ChatRenderer, TreeEpisodeResult, VerlBackend
from orchrl.trainer.mate_rollout_adapter import MateRolloutAdapter
from trajectory import BranchResult, EpisodeResult, EpisodeTrajectory, TurnData


@dataclass
class _PromptLoader:
    prompts: list[dict]

    def get_step_batch(self, *, step_idx: int, batch_size: int):
        assert step_idx == 0
        assert batch_size == 1
        return self.prompts


class _RewardProvider:
    def compute(self, trajectory):
        return {"agent_rewards": {"verifier": 1.0}, "final_reward": 1.0}


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [10, 20, len(messages)]

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


class _FakeManager:
    def __init__(self, config, server_handles):
        self.config = config
        self.server_handles = list(server_handles)

    async def generate(self, request_id, prompt_ids, sampling_params):
        class _Output:
            token_ids = [1, 2]
            log_probs = [-0.1, -0.2]
            stop_reason = "stop"
            text = "ok"
            routed_experts = None

        return _Output()


def _turn(role: str, turn_index: int, timestamp: float) -> TurnData:
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[turn_index + 1],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=timestamp,
        prompt_ids=[1000 + turn_index],
        metadata={},
    )


def _episode(episode_id: str) -> EpisodeResult:
    return EpisodeResult(
        trajectory=EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories={"verifier": [_turn("verifier", 0, 1.0)]},
            metadata={},
        ),
        rewards={"verifier": 1.0},
        final_reward=1.0,
        metadata={},
    )


def _tree_episode(episode_id: str) -> TreeEpisodeResult:
    pilot = _episode(f"{episode_id}-pilot")
    branch = _episode(f"{episode_id}-branch")
    return TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(
                episode_result=branch,
                branch_turn=0,
                branch_agent_role="verifier",
                parent_episode_id=pilot.trajectory.episode_id,
            )
        ],
        prompt="prompt",
        tree_metadata={},
    )


def _config(rollout_mode: str) -> dict:
    return {
        "roles": ["verifier"],
        "role_policy_mapping": {"verifier": "policy_a"},
        "batch_size": 1,
        "n_samples_per_prompt": 1,
        "rollout_mode": rollout_mode,
        "mas_command_template": "python fake.py --config {config_path} --question {prompt}",
        "config_template": {"llm": {"base_url": "http://placeholder/v1"}},
        "timeout": 30,
    }


def _build_adapter(monkeypatch: pytest.MonkeyPatch, rollout_mode: str) -> MateRolloutAdapter:
    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.AsyncLLMServerManager", _FakeManager)
    return MateRolloutAdapter(
        config=_config(rollout_mode),
        prompt_loader=_PromptLoader([{"prompt": "hello"}]),
        reward_provider=_RewardProvider(),
        server_address_dict={"policy_a": ["127.0.0.1:9000"]},
        server_handle_dict={"policy_a": [object()]},
        tokenizer_dict={"policy_a": _Tokenizer()},
        role_policy_mapping={"verifier": "policy_a"},
        policy_server_name_mapping={"policy_a": "policy_a"},
    )


def test_build_backend_prefers_verl_backend_and_builds_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(monkeypatch, "parallel")

    pipe_config = adapter._build_pipe_config()
    backend = adapter._build_backend(pipe_config)

    assert isinstance(backend, VerlBackend)
    assert isinstance(pipe_config.renderer["verifier"], ChatRenderer)


@pytest.mark.asyncio
async def test_collect_step_rollouts_uses_parallel_rollout_in_parallel_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parallel_rollout(**kwargs):
        assert isinstance(kwargs["backend"], VerlBackend)
        assert "verifier" in kwargs["config"].renderer
        return [_episode("parallel-episode")]

    async def fake_tree_rollout(**kwargs):
        raise AssertionError("tree_rollout should not be used in parallel mode")

    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.parallel_rollout", fake_parallel_rollout)
    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.tree_rollout", fake_tree_rollout)

    adapter = _build_adapter(monkeypatch, "parallel")
    results = await adapter.collect_step_rollouts(step_idx=0)

    assert len(results) == 1
    assert results[0].trajectory.episode_id == "parallel-episode"


@pytest.mark.asyncio
async def test_collect_step_rollouts_uses_tree_rollout_in_tree_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parallel_rollout(**kwargs):
        raise AssertionError("parallel_rollout should not be used in tree mode")

    async def fake_tree_rollout(**kwargs):
        assert isinstance(kwargs["backend"], VerlBackend)
        assert "verifier" in kwargs["config"].renderer
        return _tree_episode("tree-episode")

    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.parallel_rollout", fake_parallel_rollout)
    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.tree_rollout", fake_tree_rollout)

    adapter = _build_adapter(monkeypatch, "tree")
    results = await adapter.collect_step_rollouts(step_idx=0)

    assert len(results) == 1
    assert isinstance(results[0], TreeEpisodeResult)
    assert results[0].pilot_result.metadata["prompt_group_id"] == "prompt-0"


def test_build_adapter_rejects_unknown_backend_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchrl.trainer.mate_rollout_adapter.AsyncLLMServerManager", _FakeManager)
    config = _config("parallel")
    config["backend_mode"] = "bogus"

    with pytest.raises(ValueError, match="backend_mode"):
        MateRolloutAdapter(
            config=config,
            prompt_loader=_PromptLoader([{"prompt": "hello"}]),
            reward_provider=_RewardProvider(),
            server_address_dict={"policy_a": ["127.0.0.1:9000"]},
            server_handle_dict={"policy_a": [object()]},
            tokenizer_dict={"policy_a": _Tokenizer()},
            role_policy_mapping={"verifier": "policy_a"},
            policy_server_name_mapping={"policy_a": "policy_a"},
        )

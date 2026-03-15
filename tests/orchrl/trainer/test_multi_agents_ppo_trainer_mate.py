from __future__ import annotations

from types import SimpleNamespace

import pytest

from trajectory import BranchResult, EpisodeResult, EpisodeTrajectory, TreeEpisodeResult, TurnData

from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer


def _turn(role: str, turn_index: int, timestamp: float) -> TurnData:
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=[{"role": "user", "content": f"{role}-{turn_index}"}],
        response_text=f"{role}-response-{turn_index}",
        token_ids=[1],
        logprobs=[-0.1],
        finish_reason="stop",
        timestamp=timestamp,
        metadata={},
    )


def _episode(episode_id: str, reward: float) -> EpisodeResult:
    return EpisodeResult(
        trajectory=EpisodeTrajectory(
            episode_id=episode_id,
            agent_trajectories={
                "verifier": [_turn("verifier", 0, 1.0)],
                "searcher": [_turn("searcher", 0, 2.0)],
            },
            metadata={},
        ),
        rewards={"verifier": reward, "searcher": 0.0},
        final_reward=reward,
        metadata={"prompt_group_id": "prompt-0", "sample_idx": 0},
        status="success",
    )


def test_collect_mate_step_batches_uses_tree_adapter_when_configured(monkeypatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(max_prompt_length=16, max_response_length=16))
    trainer.ppo_trainer_dict = {
        "policy_v": SimpleNamespace(config=SimpleNamespace(data=SimpleNamespace(max_prompt_length=16, max_response_length=16))),
    }
    trainer.agent_policy_mapping = {"verifier": "policy_v"}
    trainer.mate_config = {"role_policy_mapping": {"verifier": "policy_v"}, "rollout_mode": "tree"}
    trainer.tokenizer_dict = {"policy_v": object()}

    tree_result = TreeEpisodeResult(
        pilot_result=_episode("pilot", 1.0),
        branch_results=[],
        prompt="prompt",
        tree_metadata={},
    )

    monkeypatch.setattr(trainer, "_collect_mate_episodes", lambda step_idx: [tree_result])

    called = {}

    def fake_parallel(**kwargs):
        raise AssertionError("parallel adapter should not be used")

    def fake_tree(**kwargs):
        called["tree"] = kwargs["episodes"]
        return {"policy_v": "tree-batch"}

    monkeypatch.setattr("orchrl.trainer.multi_agents_ppo_trainer.episodes_to_policy_batches", fake_parallel)
    monkeypatch.setattr("orchrl.trainer.multi_agents_ppo_trainer.tree_episodes_to_policy_batches", fake_tree)

    result = trainer._collect_mate_step_batches(step_idx=0)

    assert result == {"policy_v": "tree-batch"}
    assert called["tree"] == [tree_result]


def test_validate_flattens_tree_results_for_branch_metrics(monkeypatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.config = SimpleNamespace(training=SimpleNamespace(if_save=False))
    trainer.mate_config = {"role_policy_mapping": {"verifier": "policy_v", "searcher": "policy_s"}, "rollout_mode": "tree"}

    pilot = _episode("pilot", 1.0)
    success_branch = _episode("branch-success", 1.0)
    failed_branch = _episode("branch-failed", 0.0)
    failed_branch.status = "failed"

    tree_result = TreeEpisodeResult(
        pilot_result=pilot,
        branch_results=[
            BranchResult(
                episode_result=success_branch,
                branch_turn=0,
                branch_agent_role="verifier",
                parent_episode_id=pilot.trajectory.episode_id,
            ),
            BranchResult(
                episode_result=failed_branch,
                branch_turn=1,
                branch_agent_role="searcher",
                parent_episode_id=pilot.trajectory.episode_id,
            ),
        ],
        prompt="prompt",
        tree_metadata={},
    )

    monkeypatch.setattr(trainer, "_collect_mate_episodes", lambda step_idx: [tree_result])

    metrics = trainer._validate(global_steps=3)

    assert metrics["validation/env_state_success_rate"] == 1.0
    assert metrics["validation/agent_verifier/success_rate"] == 1.0
    assert metrics["validation/agent_verifier/avg_turns"] == 1.0


def test_require_expected_mate_policy_batches_allows_partial_policy_batches() -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.ppo_trainer_dict = {
        "verifier_model": object(),
        "searcher_model": object(),
        "answerer_model": object(),
    }

    trainer._require_expected_mate_policy_batches(
        {
            "verifier_model": object(),
            "answerer_model": object(),
        }
    )


def test_require_expected_mate_policy_batches_rejects_empty_output() -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.ppo_trainer_dict = {
        "verifier_model": object(),
        "searcher_model": object(),
        "answerer_model": object(),
    }

    with pytest.raises(RuntimeError, match="produced no policy batches"):
        trainer._require_expected_mate_policy_batches({})


def test_fit_skips_missing_policy_batches_during_collection_and_metrics(monkeypatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.config = SimpleNamespace(
        specialization="full",
        training=SimpleNamespace(
            total_training_steps=1,
            val_freq=99,
        ),
    )
    trainer.lora_differ_mode = False
    trainer.use_lora_for_generation = False
    trainer.agent_untrained = []
    trainer.ppo_trainer_dict = {
        "verifier_model": SimpleNamespace(
            _load_checkpoint=lambda: 0,
            actor_rollout_wg=SimpleNamespace(world_size=1),
            use_critic=False,
            config=SimpleNamespace(filter_ratio=0.0, filter_method="uid"),
            global_steps=0,
        ),
        "searcher_model": SimpleNamespace(
            _load_checkpoint=lambda: 0,
            actor_rollout_wg=SimpleNamespace(world_size=1),
            use_critic=False,
            config=SimpleNamespace(filter_ratio=0.0, filter_method="uid"),
            global_steps=0,
        ),
        "answerer_model": SimpleNamespace(
            _load_checkpoint=lambda: 0,
            actor_rollout_wg=SimpleNamespace(world_size=1),
            use_critic=False,
            config=SimpleNamespace(filter_ratio=0.0, filter_method="uid"),
            global_steps=0,
        ),
    }

    logged = {}
    trainer._initialize_logger_safely = lambda: SimpleNamespace(
        log=lambda data, step: logged.update({"data": data, "step": step})
    )

    verifier_batch = SimpleNamespace(batch={"tokens": [1]}, non_tensor_batch={}, meta_info={})
    answerer_batch = SimpleNamespace(batch={"tokens": [2]}, non_tensor_batch={}, meta_info={})
    trainer._collect_mate_step_batches = lambda step_idx: {
        "verifier_model": verifier_batch,
        "answerer_model": answerer_batch,
    }

    monkeypatch.setattr(trainer, "_pad_dataproto_to_world_size", lambda batch, world_size: batch)
    monkeypatch.setattr(trainer, "_finalize_batch_for_update", lambda batch, ppo_trainer: batch)

    def fake_update_parameters(batch, ppo_trainer, timing_raw):
        batch.meta_info["metrics"] = {"updated": True}
        return batch

    monkeypatch.setattr(trainer, "_update_parameters", fake_update_parameters)

    def fake_compute_data_metrics(batch, use_critic):
        if not isinstance(batch.batch, dict):
            raise AssertionError("empty placeholder batch should be skipped")
        return {"batch_seen": batch.batch["tokens"][0]}

    monkeypatch.setattr(
        "orchrl.trainer.multi_agents_ppo_trainer.compute_data_metrics",
        fake_compute_data_metrics,
    )
    monkeypatch.setattr(
        "orchrl.trainer.multi_agents_ppo_trainer.compute_timing_metrics",
        lambda batch, timing_raw: {"timing_seen": 1},
    )

    class _ProgressBar:
        def update(self, _value):
            return None

        def set_description(self, _desc):
            return None

        def close(self):
            return None

    monkeypatch.setattr("orchrl.trainer.multi_agents_ppo_trainer.tqdm", lambda *args, **kwargs: _ProgressBar())

    trainer.fit()

    assert logged["step"] == 1
    assert "verifier_model_batch_seen" in logged["data"]
    assert "answerer_model_batch_seen" in logged["data"]
    assert "searcher_model_batch_seen" not in logged["data"]

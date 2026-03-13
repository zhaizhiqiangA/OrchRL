from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import orchrl.trainer.multi_agents_ppo_trainer as trainer_module
from orchrl.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer


class FakeBatch:
    def __init__(self, name: str) -> None:
        self.name = name
        self.batch = {"payload": name}
        self.non_tensor_batch = {}
        self.meta_info = {}


class FakeLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, data, step: int) -> None:
        self.records.append((step, data))


class FakeTqdm:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def update(self, *_args, **_kwargs) -> None:
        pass

    def set_description(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def _noop_timer(*_args, **_kwargs):
    yield


def _make_policy_trainer(policy_name: str):
    return SimpleNamespace(
        name=policy_name,
        actor_rollout_wg=SimpleNamespace(world_size=1),
        use_critic=False,
        global_steps=0,
        _load_checkpoint=lambda: 0,
    )


def test_resolve_mate_policy_batches_allows_missing_single_policy() -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.ppo_trainer_dict = {"policy_a": object(), "policy_b": object()}

    present, missing = trainer._resolve_mate_policy_batches({"policy_a": object()})

    assert present == ["policy_a"]
    assert missing == ["policy_b"]


def test_resolve_mate_policy_batches_raises_when_all_batches_missing() -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.ppo_trainer_dict = {"policy_a": object()}

    with pytest.raises(RuntimeError, match="produced no policy batches"):
        trainer._resolve_mate_policy_batches({})


def test_fit_skips_updates_for_missing_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = MultiAgentsPPOTrainer.__new__(MultiAgentsPPOTrainer)
    trainer.ppo_trainer_dict = {
        "policy_a": _make_policy_trainer("policy_a"),
        "policy_b": _make_policy_trainer("policy_b"),
    }
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(
            total_training_steps=1,
            val_freq=999,
            enable_multimodal=False,
        )
    )
    trainer.global_steps = 0
    trainer.lora_differ_mode = False
    trainer.use_lora_for_generation = False

    logger = FakeLogger()
    finalized = []
    updated = []
    data_metric_batches = []
    timing_metric_batches = []
    prints = []

    trainer._initialize_logger_safely = lambda: logger
    trainer._collect_mate_step_batches = lambda step_idx: {"policy_a": FakeBatch("policy_a")}
    trainer._pad_dataproto_to_world_size = lambda batch, world_size: batch

    def fake_finalize(batch, ppo_trainer):
        finalized.append(ppo_trainer.name)
        return batch

    def fake_update(batch, ppo_trainer, timing_raw):
        updated.append(ppo_trainer.name)
        batch.meta_info = {"metrics": {"loss": 1.0}}
        batch.non_tensor_batch = {"agent_name": [ppo_trainer.name]}
        return batch

    trainer._finalize_batch_for_update = fake_finalize
    trainer._update_parameters = fake_update

    monkeypatch.setattr(trainer_module, "simple_timer", _noop_timer)
    monkeypatch.setattr(trainer_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(trainer_module, "colorful_print", lambda message, color=None: prints.append((message, color)))

    def fake_compute_data_metrics(batch, use_critic):
        data_metric_batches.append(batch.name)
        return {"num_samples": 1}

    def fake_compute_timing_metrics(batch, timing_raw):
        timing_metric_batches.append(batch.name)
        return {"step_seconds": 1.0}

    monkeypatch.setattr(trainer_module, "compute_data_metrics", fake_compute_data_metrics)
    monkeypatch.setattr(trainer_module, "compute_timing_metrics", fake_compute_timing_metrics)

    trainer.fit()

    assert finalized == ["policy_a"]
    assert updated == ["policy_a"]
    assert data_metric_batches == ["policy_a"]
    assert timing_metric_batches == ["policy_a"]
    assert logger.records

    logged_step, logged_metrics = logger.records[0]
    assert logged_step == 1
    assert logged_metrics["training/present_policy_count"] == 1
    assert logged_metrics["training/skipped_policy_count"] == 1
    assert logged_metrics["training/skipped_policies"] == "policy_b"
    assert "policy_a_num_samples" in logged_metrics
    assert "policy_b_num_samples" not in logged_metrics
    assert any("policy_b" in message for message, _ in prints)

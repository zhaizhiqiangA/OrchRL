from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import yaml


def test_search_mas_tree_real_smoke_config_has_minimal_real_smoke_settings() -> None:
    config_dir = Path("orchrl/config/search").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="search_mas_tree_real_smoke")

    repo_root = Path.cwd().resolve()

    assert cfg.specialization == "full"
    assert len(cfg.base_models) == 3
    assert len(cfg.models) == 3
    assert cfg.training.total_training_steps == 2
    assert cfg.training.val_freq == 1
    assert cfg.training.train_batch_size == 1
    assert cfg.training.train_sample_num == 1
    assert cfg.training.max_prompt_length <= 2048
    assert cfg.training.max_response_length <= 1024
    assert cfg.training.if_save is False

    assert cfg.training.mate.rollout_mode == "tree"
    assert cfg.training.mate.tree.k_branches == 1
    assert cfg.training.mate.prompt_loader.source_type == "jsonl"

    mas_work_dir = Path(cfg.training.mate.mas_work_dir)
    assert mas_work_dir.resolve() == repo_root / "examples/mas_app/search"
    assert mas_work_dir.exists(), str(mas_work_dir)

    template_path = Path(cfg.training.mate.config_template_path)
    assert template_path.resolve() == repo_root / "orchrl/config/search/templates/search_mas_tree_real_smoke_template.yaml"
    assert template_path.exists(), str(template_path)

    prompt_path = Path(cfg.training.mate.prompt_loader.path)
    assert prompt_path.resolve() == repo_root / "orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl"
    assert prompt_path.exists(), str(prompt_path)

    raw_cfg = yaml.safe_load(Path("orchrl/config/search/search_mas_tree_real_smoke.yaml").read_text())
    for model_key in ("model_0", "model_1", "model_2"):
        model_cfg = raw_cfg["models"][model_key]["ppo_trainer_config"]
        actor_model_cfg = model_cfg["actor_rollout_ref"]["model"]
        actor_cfg = model_cfg["actor_rollout_ref"]["actor"]
        rollout_cfg = model_cfg["actor_rollout_ref"]["rollout"]

        assert actor_model_cfg["use_remove_padding"] is False
        assert actor_cfg["ppo_mini_batch_size"] == 1
        assert actor_cfg["ppo_micro_batch_size"] == 1
        assert actor_cfg["ppo_micro_batch_size_per_gpu"] == 1
        assert rollout_cfg["enforce_eager"] is True
        assert rollout_cfg["max_model_len"] <= 4096

    template_cfg = yaml.safe_load(template_path.read_text())
    assert template_cfg["application"]["max_turns"] == 4
    assert template_cfg["application"]["force_final_answer_on_max_turn"] is True

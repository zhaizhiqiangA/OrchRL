from __future__ import annotations

import pytest

from orchrl.trainer.mate_config import validate_mate_config


def _base_config() -> dict:
    return {
        "roles": ["verifier", "searcher"],
        "role_policy_mapping": {
            "verifier": "policy_a",
            "searcher": "policy_b",
        },
    }


def test_validate_mate_config_defaults_rollout_mode_to_parallel() -> None:
    config = validate_mate_config(
        _base_config(),
        {"verifier": "policy_a", "searcher": "policy_b"},
    )

    assert config["rollout_mode"] == "parallel"


def test_validate_mate_config_accepts_tree_rollout_mode() -> None:
    config = _base_config()
    config["rollout_mode"] = "tree"

    validated = validate_mate_config(
        config,
        {"verifier": "policy_a", "searcher": "policy_b"},
    )

    assert validated["rollout_mode"] == "tree"


def test_validate_mate_config_rejects_unknown_rollout_mode() -> None:
    config = _base_config()
    config["rollout_mode"] = "bogus"

    with pytest.raises(ValueError, match="rollout_mode"):
        validate_mate_config(
            config,
            {"verifier": "policy_a", "searcher": "policy_b"},
        )


def test_validate_mate_config_rejects_invalid_tree_branch_settings() -> None:
    config = _base_config()
    config["rollout_mode"] = "tree"
    config["tree"] = {"k_branches": 0}

    with pytest.raises(ValueError, match="k_branches"):
        validate_mate_config(
            config,
            {"verifier": "policy_a", "searcher": "policy_b"},
        )

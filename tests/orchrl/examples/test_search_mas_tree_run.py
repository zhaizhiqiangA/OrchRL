from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path.cwd().resolve()
    script_path = repo_root / "examples/search_mas_tree/run.py"
    spec = importlib.util.spec_from_file_location("search_mas_tree_smoke", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_smoke_runtime_resolves_repo_local_paths() -> None:
    module = _load_module()
    runtime = module.load_smoke_runtime()

    repo_root = Path.cwd().resolve()

    assert runtime.repo_root == repo_root
    assert runtime.config_name == "search_mas_tree_real_smoke"
    assert runtime.config_dir == repo_root / "orchrl/config/search"
    assert runtime.mas_work_dir == repo_root / "examples/mas_app/search"
    assert runtime.config_template_path == repo_root / "orchrl/config/search/templates/search_mas_tree_real_smoke_template.yaml"
    assert runtime.prompt_data_path == repo_root / "orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl"
    assert len(runtime.model_paths) == 3
    assert all(isinstance(path, Path) for path in runtime.model_paths)
    assert all(path.name for path in runtime.model_paths)


def test_build_smoke_env_sets_required_overrides(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SEARCH_MAS_LLM_BASE_URL", "http://bad.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://bad.example/v1")
    monkeypatch.setenv("LIBRARY_PATH", "/tmp/original_lib")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/original_ld")

    env = module.build_smoke_env(
        retrieval_service_url="http://127.0.0.1:8010/retrieve",
        cuda_visible_devices="0,1,2",
    )

    assert "SEARCH_MAS_LLM_BASE_URL" not in env
    assert "OPENAI_BASE_URL" not in env
    assert env["SEARCH_MAS_RETRIEVAL_SERVICE_URL"] == "http://127.0.0.1:8010/retrieve"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["VLLM_USE_V1"] == "1"
    assert env["WANDB_MODE"] == "offline"
    assert env["HYDRA_FULL_ERROR"] == "1"
    assert env["LIBRARY_PATH"] == "/usr/local/cuda/compat:/usr/local/cuda/targets/x86_64-linux/lib/stubs:/tmp/original_lib"
    assert env["LD_LIBRARY_PATH"] == "/tmp/original_ld"


def test_build_train_command_targets_orchrl_trainer_entrypoint() -> None:
    module = _load_module()
    runtime = module.load_smoke_runtime()

    command = module.build_train_command(runtime)

    assert command == [
        "python3",
        "-m",
        "orchrl.trainer.train",
        "--config-path",
        str(runtime.config_dir),
        "--config-name",
        runtime.config_name,
    ]


def test_probe_retrieval_service_uses_relaxed_default_timeout(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, float] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)

    module.probe_retrieval_service("http://127.0.0.1:8010/retrieve")

    assert module.DEFAULT_RETRIEVAL_TIMEOUT_SEC == 30.0
    assert captured["timeout"] == module.DEFAULT_RETRIEVAL_TIMEOUT_SEC

#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence
from urllib import error, request

from hydra import compose, initialize_config_dir

DEFAULT_RETRIEVAL_TIMEOUT_SEC = 30.0
CUDA_COMPAT_PATH = "/usr/local/cuda/compat"
CUDA_STUBS_PATH = "/usr/local/cuda/targets/x86_64-linux/lib/stubs"


@dataclass(frozen=True)
class SmokeRuntime:
    repo_root: Path
    config_name: str
    config_dir: Path
    mas_work_dir: Path
    config_template_path: Path
    prompt_data_path: Path
    model_paths: tuple[Path, ...]


def load_smoke_runtime(config_name: str = "search_mas_tree_real_smoke") -> SmokeRuntime:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "orchrl/config/search"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    model_paths = tuple(Path(str(model_cfg.path)) for model_cfg in cfg.base_models.values())
    return SmokeRuntime(
        repo_root=repo_root,
        config_name=config_name,
        config_dir=config_dir,
        mas_work_dir=(repo_root / str(cfg.training.mate.mas_work_dir)).resolve(),
        config_template_path=(repo_root / str(cfg.training.mate.config_template_path)).resolve(),
        prompt_data_path=(repo_root / str(cfg.training.mate.prompt_loader.path)).resolve(),
        model_paths=model_paths,
    )


def _prepend_env_path(existing_value: str | None, prefixes: Sequence[str]) -> str:
    parts = [prefix for prefix in prefixes if prefix]
    if existing_value:
        parts.append(existing_value)
    return ":".join(parts)


def build_smoke_env(
    *,
    repo_root: Path | None = None,
    retrieval_service_url: str,
    cuda_visible_devices: str,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ if base_env is None else base_env)
    env.pop("SEARCH_MAS_LLM_BASE_URL", None)
    env.pop("OPENAI_BASE_URL", None)
    env["SEARCH_MAS_RETRIEVAL_SERVICE_URL"] = retrieval_service_url
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["VLLM_USE_V1"] = "1"
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    env["REPO_ROOT"] = str(repo_root)
    env["PYTHONPATH"] = _prepend_env_path(
        env.get("PYTHONPATH"),
        [str(repo_root), str(repo_root / "verl")],
    )
    env["LIBRARY_PATH"] = _prepend_env_path(
        env.get("LIBRARY_PATH"),
        [CUDA_COMPAT_PATH, CUDA_STUBS_PATH],
    )
    return env


def build_train_command(runtime: SmokeRuntime) -> list[str]:
    return [
        "python3",
        "-m",
        "orchrl.trainer.train",
        "--config-path",
        str(runtime.config_dir),
        "--config-name",
        runtime.config_name,
    ]


def validate_runtime_paths(runtime: SmokeRuntime) -> None:
    required_paths: Sequence[Path] = (
        runtime.mas_work_dir,
        runtime.config_template_path,
        runtime.prompt_data_path,
        *runtime.model_paths,
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing smoke runtime paths:\n{joined}")


def probe_retrieval_service(
    retrieval_service_url: str,
    timeout_sec: float = DEFAULT_RETRIEVAL_TIMEOUT_SEC,
) -> None:
    payload = json.dumps({"query": "healthcheck", "topk": 1}).encode("utf-8")
    req = request.Request(
        retrieval_service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                raise RuntimeError(f"retrieval service returned status {resp.status}")
    except error.URLError as exc:
        raise RuntimeError(f"retrieval service probe failed: {retrieval_service_url}") from exc


def run_smoke(
    *,
    runtime: SmokeRuntime,
    retrieval_service_url: str,
    cuda_visible_devices: str,
    log_path: Path,
    check_retrieval: bool,
) -> int:
    validate_runtime_paths(runtime)
    if check_retrieval:
        probe_retrieval_service(retrieval_service_url)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = build_smoke_env(
        repo_root=runtime.repo_root,
        retrieval_service_url=retrieval_service_url,
        cuda_visible_devices=cuda_visible_devices,
    )
    command = build_train_command(runtime)

    print(f"[INFO] Repo root: {runtime.repo_root}")
    print(f"[INFO] Config: {runtime.config_name}")
    print(f"[INFO] Config dir: {runtime.config_dir}")
    print(f"[INFO] MAS work dir: {runtime.mas_work_dir}")
    print(f"[INFO] Template path: {runtime.config_template_path}")
    print(f"[INFO] Prompt data: {runtime.prompt_data_path}")
    print(f"[INFO] Model paths: {' | '.join(str(path) for path in runtime.model_paths)}")
    print(f"[INFO] Retrieval URL: {retrieval_service_url}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    print(f"[INFO] Log path: {log_path}")
    print(f"[INFO] Command: {' '.join(command)}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=runtime.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Search MAS tree real smoke.")
    parser.add_argument("--config-name", default="search_mas_tree_real_smoke")
    parser.add_argument("--retrieval-service-url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--cuda-visible-devices", default="0,1,2")
    parser.add_argument("--log-path", default="logs/search_mas_tree_real_smoke.log")
    parser.add_argument("--skip-retrieval-healthcheck", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    runtime = load_smoke_runtime(config_name=args.config_name)
    validate_runtime_paths(runtime)
    if not args.skip_retrieval_healthcheck:
        probe_retrieval_service(args.retrieval_service_url)

    print(f"[INFO] Repo root: {runtime.repo_root}")
    print(f"[INFO] Config: {runtime.config_name}")
    print(f"[INFO] Config dir: {runtime.config_dir}")
    print(f"[INFO] MAS work dir: {runtime.mas_work_dir}")
    print(f"[INFO] Template path: {runtime.config_template_path}")
    print(f"[INFO] Prompt data: {runtime.prompt_data_path}")
    print(f"[INFO] Model paths: {' | '.join(str(path) for path in runtime.model_paths)}")
    print(f"[INFO] Retrieval URL: {args.retrieval_service_url}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")
    print(f"[INFO] Log path: {Path(args.log_path).resolve()}")

    if args.check_only:
        return 0

    return run_smoke(
        runtime=runtime,
        retrieval_service_url=args.retrieval_service_url,
        cuda_visible_devices=args.cuda_visible_devices,
        log_path=Path(args.log_path).resolve(),
        check_retrieval=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())

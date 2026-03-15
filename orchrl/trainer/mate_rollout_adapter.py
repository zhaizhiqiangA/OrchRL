from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import yaml

from trajectory import (
    AgentPipeConfig,
    ModelMappingEntry,
    TreeEpisodeResult,
    VLLMBackend,
    parallel_rollout,
    tree_rollout,
)


def _to_plain_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("mate rollout config must resolve to a dict")
        return resolved
    raise TypeError("mate rollout config must be a dict")


class _JobAwareRewardProvider:
    def __init__(self, reward_provider, metadata: dict[str, Any]):
        self._reward_provider = reward_provider
        self._metadata = metadata

    def compute(self, trajectory):
        trajectory.metadata.update(self._metadata)
        return self._reward_provider.compute(trajectory)


class MateRolloutAdapter:
    def __init__(
        self,
        *,
        config,
        prompt_loader,
        reward_provider,
        server_address_dict,
        role_policy_mapping,
        policy_server_name_mapping,
    ):
        self._config = _to_plain_dict(config)
        self._prompt_loader = prompt_loader
        self._reward_provider = reward_provider
        self._server_address_dict = server_address_dict
        self._role_policy_mapping = dict(role_policy_mapping)
        self._policy_server_name_mapping = dict(policy_server_name_mapping)
        self._roles = list(self._config.get("roles", self._role_policy_mapping.keys()))
        sampling_cfg = self._config.get("sampling", {})
        self._batch_size = int(self._config.get("batch_size", sampling_cfg.get("n_prompts_per_step", 1)))
        self._n_samples_per_prompt = int(self._config.get("n_samples_per_prompt", sampling_cfg.get("n_samples_per_prompt", 1)))
        max_concurrent = self._config.get("max_concurrent_episodes", sampling_cfg.get("max_concurrent_episodes"))
        self._max_concurrent_episodes = int(max_concurrent) if max_concurrent is not None else None
        self._rollout_mode = str(self._config.get("rollout_mode", "parallel"))
        tree_cfg = self._config.get("tree", {})
        k_branches = tree_cfg.get("k_branches", self._config.get("k_branches", 1))
        max_concurrent_branches = tree_cfg.get(
            "max_concurrent_branches",
            self._config.get("max_concurrent_branches"),
        )
        self._k_branches = int(k_branches)
        self._max_concurrent_branches = (
            int(max_concurrent_branches) if max_concurrent_branches is not None else None
        )

    async def collect_step_rollouts(self, step_idx: int):
        prompts = self._prompt_loader.get_step_batch(step_idx=step_idx, batch_size=self._batch_size)
        if not prompts:
            return []

        pipe_config = self._build_pipe_config()
        backend = self._build_backend(pipe_config)
        jobs = [
            {
                "prompt_item": prompt_item,
                "prompt_group_id": f"prompt-{prompt_idx}",
                "sample_idx": sample_idx,
            }
            for prompt_idx, prompt_item in enumerate(prompts)
            for sample_idx in range(self._n_samples_per_prompt)
        ]
        semaphore = asyncio.Semaphore(self._max_concurrent_episodes) if self._max_concurrent_episodes is not None else None

        async def run_job(job):
            if semaphore is None:
                return await self._collect_single_job(job=job, pipe_config=pipe_config, backend=backend)
            async with semaphore:
                return await self._collect_single_job(job=job, pipe_config=pipe_config, backend=backend)

        gathered = await asyncio.gather(*(run_job(job) for job in jobs))
        episodes = []
        for results in gathered:
            episodes.extend(results)
        return episodes

    async def _collect_single_job(self, *, job, pipe_config: AgentPipeConfig, backend: VLLMBackend):
        job_metadata = {
            "prompt": job["prompt_item"]["prompt"],
            "expected": job["prompt_item"].get("expected"),
            "prompt_row": job["prompt_item"].get("raw"),
            "prompt_group_id": job["prompt_group_id"],
            "sample_idx": job["sample_idx"],
        }
        reward_provider = _JobAwareRewardProvider(self._reward_provider, job_metadata)
        if self._rollout_mode == "tree":
            result = await tree_rollout(
                prompt=job["prompt_item"]["prompt"],
                reward_provider=reward_provider,
                config=pipe_config,
                backend=backend,
                k_branches=self._k_branches,
                max_concurrent_branches=self._max_concurrent_branches,
            )
            self._annotate_tree_result(result, job_metadata)
            return [result]

        results = await parallel_rollout(
            prompts=[job["prompt_item"]["prompt"]],
            reward_provider=reward_provider,
            config=pipe_config,
            backend=backend,
            n_samples_per_prompt=1,
            max_concurrent=None,
        )
        for result in results:
            result.metadata.update(job_metadata)
            result.trajectory.metadata.update(job_metadata)
        return results

    def _build_pipe_config(self) -> AgentPipeConfig:
        model_mapping: dict[str, ModelMappingEntry] = {}
        for role in self._roles:
            policy_name = self._role_policy_mapping[role]
            backend_url = self._select_backend_url(policy_name)
            actual_model = self._policy_server_name_mapping.get(policy_name, policy_name)
            model_mapping[role] = ModelMappingEntry(
                actual_model=actual_model,
                backend_url=backend_url,
            )

        return AgentPipeConfig(
            mas_command_template=self._config["mas_command_template"],
            config_template=self._load_config_template(),
            model_mapping=model_mapping,
            timeout=float(self._config.get("timeout", 300.0)),
            monitor_host=self._config.get("monitor_host", "127.0.0.1"),
            monitor_port=int(self._config.get("monitor_port", 0)),
            mas_work_dir=Path(self._config["mas_work_dir"]) if self._config.get("mas_work_dir") else None,
        )

    def _build_backend(self, pipe_config: AgentPipeConfig) -> VLLMBackend:
        default_url = next(
            (entry.backend_url for entry in pipe_config.model_mapping.values() if entry.backend_url),
            None,
        )
        if default_url is None:
            raise ValueError("no backend_url available for MATE rollout backend")
        return VLLMBackend(
            backend_url=default_url,
            timeout=float(self._config.get("backend_timeout", self._config.get("timeout", 120.0))),
        )

    def _select_backend_url(self, policy_name: str) -> str:
        addresses = self._server_address_dict.get(policy_name)
        if isinstance(addresses, (list, tuple)):
            backend_url = addresses[0] if addresses else None
        else:
            backend_url = addresses
        if not isinstance(backend_url, str) or not backend_url:
            raise ValueError(f"No server address configured for policy '{policy_name}'")
        return self._normalize_backend_url(backend_url)

    @staticmethod
    def _normalize_backend_url(backend_url: str) -> str:
        normalized = backend_url.rstrip("/")
        if normalized.startswith(("http://", "https://")):
            return normalized
        return f"http://{normalized}"

    def _load_config_template(self) -> dict[str, Any]:
        inline_template = self._config.get("config_template")
        if isinstance(inline_template, dict):
            return dict(inline_template)

        template_path = self._config.get("config_template_path")
        if not isinstance(template_path, str) or not template_path:
            raise ValueError("mate config requires either config_template or config_template_path")

        with open(template_path, "r", encoding="utf-8") as file_obj:
            loaded = yaml.safe_load(file_obj)
        if not isinstance(loaded, dict):
            raise ValueError("mate config template must load as a dict")
        return loaded

    @staticmethod
    def _annotate_tree_result(result: TreeEpisodeResult, metadata: dict[str, Any]) -> None:
        def annotate_episode(episode_result) -> None:
            episode_result.metadata.update(metadata)
            episode_result.trajectory.metadata.update(metadata)

        annotate_episode(result.pilot_result)
        for branch in result.branch_results:
            annotate_episode(branch.episode_result)

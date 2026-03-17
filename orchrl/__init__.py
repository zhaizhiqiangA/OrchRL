# Monkey patch a few functions to avoid name collision with multi trainers. In the long run, these changes should be merged to verl.

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
from typing import Optional, Any
import asyncio
import ray
from verl.utils.net_utils import is_valid_ipv6_address
from verl.utils.device import get_resource_name


async def launch_servers(self):
    """Launch http server in each node."""
    assert len(self.workers) == self.world_size, (
        f"worker number {len(self.workers)} not equal to world size {self.world_size}"
    )

    # NOTE: We always use MP Executor backend whether it's single-node or multi-node.
    # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
    if self.config.data_parallel_size == 1 and self.nnodes > 1:
        assert _VLLM_VERSION >= version.parse("0.11.1"), (
            "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
        )

    # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                )
            )
            for worker in self.workers
        ]
    )
    worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
    worker_node_ids = [worker_info[0] for worker_info in worker_infos]

    # create server actor in each node with node affinity and cuda visible devices
    nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
    for node_rank in range(nnodes):
        workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
        node_cuda_visible_devices = ",".join(
            worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
        )
        node_id = worker_node_ids[node_rank * gpus_per_replica_node]
        name = (
            f"vllm_server_{self.replica_rank}_{node_rank}"
            if not self.is_reward_model
            else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
        )
        # append model_id to server name if model_id is specified in model_config
        if 'override_config' in self.model_config and 'model_id' in self.model_config.override_config:
            name += f'_{self.model_config.override_config["model_id"]}'
        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            name=name,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=node_rank,
            gpus_per_node=gpus_per_replica_node,
            nnodes=nnodes,
            cuda_visible_devices=node_cuda_visible_devices,
        )
        self.servers.append(server)

    # launch http server in each node
    master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
    await asyncio.gather(
        *[
            server.launch_server.remote(
                master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
            )
            for server in self.servers
        ]
    )

    # get http server address from first server
    server_address, server_port = await self.servers[0].get_server_address.remote()
    self._server_handle = self.servers[0]
    self._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


async def _execute_method(
    self,
    method: str,
    non_block: bool = False,
    timeout: Optional[float] = None,
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> Any:
    """Execute method on inference engine via ray.

    Args:
        method: The method name to execute on the server.
        non_block: If True, execute the method asynchronously and return immediately.
        timeout: Timeout for the collective_rpc call.
        args: Positional arguments for the method.
        kwargs: Keyword arguments for the method.

    Returns:
        The result of the method execution, or None if non_block=True.
    """
    if self.rollout_rank != 0:
        return None

    # Lazy init http server adapter because http server is launched after hybrid engine.
    name = f"vllm_server_{self.replica_rank}_{self.node_rank}"
    if 'override_config' in self.model_config and 'model_id' in self.model_config.override_config:
        name += f'_{self.model_config.override_config["model_id"]}'
    if self.server_handle is None:
        self.server_handle = ray.get_actor(name)

    future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
    return future if non_block else await future

vLLMReplica.launch_servers = launch_servers
ServerAdapter._execute_method = _execute_method    
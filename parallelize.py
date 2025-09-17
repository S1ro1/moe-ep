import os
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import distribute_tensor, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module


def construct_mesh(ep_degree):
    if "RANK" not in os.environ:
        return None

    world_size = int(os.environ["WORLD_SIZE"])

    if ep_degree == 1:
        # only data parallelism
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp_shard_cp",),
        )
        return mesh

    mp_expert_subgroup = world_size // ep_degree  # assuming full ep
    # dp_shard_mod_ep = 1
    dp_shard_2_ep = world_size // 1  # assuming full ep
    # dp_shard_in_ep = 8

    mesh = init_device_mesh(
        "cuda",
        (mp_expert_subgroup, dp_shard_2_ep),
        mesh_dim_names=("dp_shard_mod_ep", "dp_shard_in_ep"),
    )
    print(f"Initialized device mesh: {mesh}")

    dp_shard_mesh_dims = ["dp_shard_mod_ep", "dp_shard_in_ep"]
    ep_mesh_dims = ["dp_shard_in_ep"]

    mesh[tuple(dp_shard_mesh_dims)]._flatten("dp_shard_cp")
    mesh[tuple(ep_mesh_dims)]._flatten("ep")
    return mesh


class Ep2DpParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self._num_tokens_to_send = None
        self._num_tokens_to_recv = None
        self._reshuffle_indices = None
        self._reshuffled_counts = None

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        ep_size = device_mesh.shape[0]

        # num_tokens_per_expert is of shape (num_experts, ), where each element holds the amount of tokens for
        # the corresponding expert from the local rank
        with torch.no_grad():
            # we transpose num_tokens_per_expert on device_mesh ep axis, to get the number of tokens for the local rank
            # think of all2all as a transpose operation on the device mesh
            # grouped_tokens_per_rank is of shape (ep_size * num_experts_per_rank,)
            # such as:
            # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ..., # tokens for local expert n from EP rank 0, ...]
            grouped_tokens_per_rank = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )

            # this is of shape (ep_size, )
            # [#tokens for rank 0, #tokens for rank 1, ...]
            num_tokens_to_send = (
                num_tokens_per_expert.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )

            # this is of shape (ep_size, )
            # [#tokens from rank 0, #tokens from rank 1, ...]
            num_tokens_to_recv = (
                grouped_tokens_per_rank.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self._num_tokens_to_send = num_tokens_to_send.tolist()
            self._num_tokens_to_recv = num_tokens_to_recv.tolist()

        # perform all-to-all to send the tokens to the right ranks
        routed_input = all_to_all_single_autograd(
            routed_input,
            self._num_tokens_to_recv,
            self._num_tokens_to_send,
            device_mesh.get_group(),
        )

        # routed input is not sorted by expert anymore, rather looks like:
        # [tokens for local expert 0 from EP rank 0, tokens for local expert 0 from EP rank 1, ..., tokens for local expert 0 from EP rank n, ...]
        # this needs to be reshuffled back
        # same applies to grouped_tokens_per_rank
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 0 from EP rank 1, ..., # tokens for local expert 0 from EP rank n, ...]
        return routed_input, grouped_tokens_per_rank

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    def _token_combine(self, mod, routed_output, device_mesh):
        # reverse all-to-all from dispatch
        routed_output = all_to_all_single_autograd(
            routed_output,
            self._num_tokens_to_send,
            self._num_tokens_to_recv,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=Ep2DpParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def apply_fsdp(
    model: torch.nn.Module,
    dp_mesh: DeviceMesh = None,
    dp_mod_ep_mesh: DeviceMesh = None,
    ep_degree: int = 1,
) -> torch.nn.Module:
    if "RANK" not in os.environ:
        return

    kwargs = {
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        "mesh": dp_mesh,
    }

    for i, layer in enumerate(model.model.layers):
        if ep_degree > 1:
            ep_kwargs = {
                **kwargs,
                "mesh": dp_mod_ep_mesh,
            }
            # shard the mlp (EP) to apply MP
            fully_shard(layer.mlp.experts, **ep_kwargs)
            # while we are sharding across a different axis, which might be of size 1 only,
            # we need to make sure the gradients are properly averaged across all the EP ranks, as ep ranks
            # basically steal from the original fsdp_ranks
            layer.mlp.experts.set_gradient_divide_factor(ep_degree)

        # shard the rest of the layer with classical FSDP
        fully_shard(layer, **kwargs)

    fully_shard(model, **kwargs)

    return model


def apply_ep(
    model,
    ep_mesh: DeviceMesh,
):
    for i, layer in enumerate(model.model.layers):
        parallelize_module(
            module=layer.mlp.experts,
            device_mesh=ep_mesh,
            parallelize_plan=Ep2DpParallel(),
        )

    return model


def parallelize_model(model: nn.Module) -> nn.Module:
    ep_degree = int(os.environ.get("EP_SIZE", 1))
    mesh = construct_mesh(ep_degree)
    if mesh is None:
        return model

    if ep_degree > 1:
        model = apply_ep(model, mesh["ep"])

    model = apply_fsdp(
        model,
        dp_mesh=mesh[("dp_shard_cp",)],
        dp_mod_ep_mesh=mesh[("dp_shard_mod_ep",)] if ep_degree > 1 else None,
        ep_degree=ep_degree,
    )

    return model

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
        print(f"Initialized device mesh: {mesh}")
        return mesh

    dp_shard_mod_ep = world_size // ep_degree  # assuming full ep
    # dp_shard_mod_ep = 1
    dp_shard_in_ep = world_size // 1  # assuming full ep
    # dp_shard_in_ep = 8

    mesh = init_device_mesh(
        "cuda",
        (dp_shard_mod_ep, dp_shard_in_ep),
        mesh_dim_names=("dp_shard_mod_ep", "dp_shard_in_ep"),
    )
    print(f"Initialized device mesh: {mesh}")

    dp_shard_mesh_dims = ["dp_shard_mod_ep", "dp_shard_in_ep"]
    ep_mesh_dims = ["dp_shard_in_ep"]

    mesh[tuple(dp_shard_mesh_dims)]._flatten("dp_shard_cp")
    mesh[tuple(ep_mesh_dims)]._flatten("ep")
    return mesh


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_size = device_mesh.shape[0]

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct format -- this is done via the function
        # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
        # each expert gets locally is a multiple of ALIGN_SIZE_M.

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
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
            fully_shard(layer.mlp, **ep_kwargs)
            layer.mlp.set_gradient_divide_factor(ep_degree)
            # shard the whole decoder laye
        fully_shard(layer, **kwargs)

    fully_shard(model, **kwargs)

    return model


def apply_ep(
    model,
    ep_mesh: DeviceMesh,
):
    for i, layer in enumerate(model.model.layers):
        parallelize_module(
            module=layer.mlp,
            device_mesh=ep_mesh,
            parallelize_plan=ExpertParallel(),
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

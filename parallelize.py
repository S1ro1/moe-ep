import os
import torch

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh


def apply_fsdp(model: torch.nn.Module, mesh: DeviceMesh | None) -> torch.nn.Module:
    if "RANK" in os.environ:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

        for i, layer in enumerate(model.model.layers):
            reshard_after_forward = i != len(model.model.layers) - 1
            fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)

        fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)

    return model

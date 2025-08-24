import torch

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh


def apply_fsdp(model: torch.nn.Module, mesh: DeviceMesh | None) -> torch.nn.Module:
    mp_policy = MixedPrecisionPolicy()

    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy, mesh=mesh)

    fully_shard(model, mp_policy=mp_policy, mesh=mesh)

    return model

from dataclasses import asdict

import os
import torch
import torch.distributed as dist
import wandb

from transformers.models import Qwen3MoeForCausalLM, Qwen3MoeConfig

from parallelize import apply_fsdp
from tracker import PerformanceLogger
from utils import TrainConfig, maybe_wandb_log, setup_distributed, destroy_distributed, parse_args


def get_model_flavour(tcfg: TrainConfig) -> Qwen3MoeForCausalLM:
    if tcfg.model_flavour == "30b":
        config = Qwen3MoeConfig(use_cache=False)
    elif tcfg.model_flavour == "debug":
        config = Qwen3MoeConfig(
            hidden_size=256,
            # intermediate_size=768,
            num_hidden_layers=4,
            # num_attention_heads=8,
            # num_key_value_heads=1,
            use_cache=False,
            num_experts=32,
            # num_experts_per_tok=2,
            moe_intermediate_size=256,
        )
    else:
        raise ValueError(f"Model flavour {tcfg.model_flavour} not supported")

    return Qwen3MoeForCausalLM(config)




def main(tcfg: TrainConfig):
    model = get_model_flavour(tcfg)
    model = apply_fsdp(model, mesh=None)
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    logger = PerformanceLogger()

    if tcfg.run_name and dist.get_rank() == 0:
        wandb.init(
            project="Qwen3-moe",
            name=tcfg.run_name,
            config={
                **asdict(tcfg),
                **model.config.__dict__,
                "use_new_moe": os.environ.get("USE_NEW_MOE", "false"),
            },
        )

    for _ in range(tcfg.num_steps):
        input_ids = torch.ones(
            (tcfg.batch_size, tcfg.sequence_length), dtype=torch.int64
        ).cuda()
        labels = input_ids.clone()
        optim.zero_grad()
        inputs = {"input_ids": input_ids, "labels": labels}
        outputs = model(**inputs)

        loss = outputs.loss
        maybe_wandb_log(tcfg, {"loss": loss.item()})
        loss.backward()
        logger.mark_step(input_ids)

        optim.step()

    maybe_wandb_log(tcfg, logger.get_metrics())


if __name__ == "__main__":
    setup_distributed()

    tcfg = parse_args()
    main(tcfg)

    destroy_distributed()

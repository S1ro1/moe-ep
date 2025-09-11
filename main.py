from dataclasses import asdict

import os
import torch
import torch.distributed as dist
import wandb

from parallelize import apply_fsdp
from tracker import PerformanceLogger
from utils import (
    TrainConfig,
    maybe_wandb_log,
    setup_distributed,
    destroy_distributed,
    parse_args,
    set_seed,
    prepare_dataloader,
    print0,
    get_model_flavour,
)


def main(tcfg: TrainConfig):
    set_seed(1)
    model = get_model_flavour(tcfg)
    model = apply_fsdp(model, mesh=None)
    optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    logger = PerformanceLogger()

    if tcfg.run_name and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(
            project="Qwen3-moe",
            name=tcfg.run_name,
            config={
                **asdict(tcfg),
                **model.config.__dict__,
                "use_new_moe": os.environ.get("USE_NEW_MOE", "false"),
            },
        )

    dataloader = prepare_dataloader(tcfg, "Qwen/Qwen3-30B-A3B")

    for step, batch in enumerate(iter(dataloader)):
        if step >= tcfg.num_steps:
            break

        optim.zero_grad()
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        print0(f"Step {step} Loss: {loss.item():.4f}")
        maybe_wandb_log(tcfg, {"loss": loss.item()})
        loss.backward()
        logger.mark_step(batch["input_ids"])
        optim.step()

    maybe_wandb_log(tcfg, logger.get_metrics())


if __name__ == "__main__":
    if "RANK" in os.environ:
        setup_distributed()

    tcfg = parse_args()
    main(tcfg)

    if "RANK" in os.environ:
        destroy_distributed()

from dataclasses import asdict

import os
import torch
import torch.distributed as dist
import wandb

from parallelize import parallelize_model
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

# from transformers import Qwen3MoeForCausalLM


def main(tcfg: TrainConfig):
    set_seed(1)
    model = get_model_flavour(tcfg)
    model = parallelize_model(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-5)
    logger = PerformanceLogger()

    run_name = f"{tcfg.model_flavour}-bs{tcfg.batch_size}-sl{tcfg.sequence_length}-ep{os.environ.get('EP_SIZE', '1')}-fsdp{dist.get_world_size()}"

    if tcfg.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(
            project="Qwen3-moe",
            name=run_name,
            config={
                **asdict(tcfg),
                **model.config.__dict__,
                "moe_forward_fn": os.environ.get(
                    "QWEN3_MOE_EXPERTS_FORWARD", "torch_grouped_mm"
                ),
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
        loss.backward()
        metrics = logger.mark_step(batch["input_ids"])
        maybe_wandb_log(tcfg, {"loss": loss.item(), **metrics})
        optim.step()

    maybe_wandb_log(tcfg, logger.get_metrics())
    # model.save_pretrained(f"./checkpoints/{run_name}")


if __name__ == "__main__":
    if "RANK" in os.environ:
        setup_distributed()

    tcfg = parse_args()
    main(tcfg)

    if "RANK" in os.environ:
        destroy_distributed()

import torch
from argparse import ArgumentParser
import wandb
import torch.distributed as dist

from typing import Literal
from dataclasses import dataclass

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def destroy_distributed():
    dist.destroy_process_group()

def print0(msg):
    if dist.get_rank() == 0:
        print(msg)


def maybe_wandb_log(tcfg, metrics: dict):
    if dist.get_rank() == 0 and tcfg.run_name:
        wandb.log(metrics)


@dataclass
class TrainConfig:
    model_flavour: Literal["debug", "30b"]

    sequence_length: int
    batch_size: int
    num_steps: int

    run_name: str


def parse_args():
    torch._grouped_mm
    parser = ArgumentParser()
    parser.add_argument("--model-flavour", default="debug", choices=["debug", "30b"])
    parser.add_argument("--sequence-length", default=1024, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-steps", default=100, type=int)
    parser.add_argument("--run-name", default=None, type=str)

    args = parser.parse_args()

    return TrainConfig(**vars(args))

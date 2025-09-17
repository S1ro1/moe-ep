import os
import torch
import numpy as np
import random
from argparse import ArgumentParser
import wandb
import torch.distributed as dist

from typing import Literal
from dataclasses import dataclass

from transformers import AutoTokenizer, Qwen3MoeForCausalLM, Qwen3MoeConfig
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def destroy_distributed():
    dist.destroy_process_group()


def print0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


def maybe_wandb_log(tcfg, metrics: dict):
    if (not dist.is_initialized() or dist.get_rank() == 0) and tcfg.run_name:
        wandb.log(metrics)


@dataclass
class TrainConfig:
    model_flavour: Literal["debug", "30b"]

    sequence_length: int
    batch_size: int
    num_steps: int

    run_name: str

    from_pretrained: bool = False
    from_old_checkpoint: bool = True


def parse_args():
    torch._grouped_mm
    parser = ArgumentParser()
    parser.add_argument("--model-flavour", default="debug", choices=["debug", "30b"])
    parser.add_argument("--sequence-length", default=1024, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-steps", default=100, type=int)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--from-pretrained", action="store_true")
    parser.add_argument("--from-old-checkpoint", action="store_true", default=True)

    args = parser.parse_args()

    return TrainConfig(**vars(args))


def prepare_dataloader(
    tcfg: TrainConfig, model_name: str
) -> torch.utils.data.DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")
    if "RANK" in os.environ:
        raw_dataset = split_dataset_by_node(
            raw_dataset, rank=dist.get_rank(), world_size=dist.get_world_size()
        )

    def tokenize_function(examples):
        tokenized_batch = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=tcfg.sequence_length,
            return_tensors=None,
        )
        tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
        return tokenized_batch

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    def create_packed_sequences(examples):
        all_tokens = []
        for input_ids in examples["input_ids"]:
            all_tokens.extend(input_ids)

        num_sequences = len(all_tokens) // (tcfg.sequence_length + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (tcfg.sequence_length + 1)
            end_idx = start_idx + (tcfg.sequence_length + 1)
            full_sequence = all_tokens[start_idx:end_idx]
            packed_input_ids.append(full_sequence[:-1])
            packed_labels.append(full_sequence[1:])

        return {"input_ids": packed_input_ids, "shift_labels": packed_labels}

    packed_dataset = tokenized_dataset.map(
        create_packed_sequences,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        batch_size=1000,
    )

    def collate_fn(batch):
        input_ids = torch.tensor(
            [item["input_ids"] for item in batch], dtype=torch.long
        )
        shift_labels = torch.tensor(
            [item["shift_labels"] for item in batch], dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "shift_labels": shift_labels,
            "labels": shift_labels,
        }

    dataloader = torch.utils.data.DataLoader(
        packed_dataset,
        batch_size=tcfg.batch_size,
        collate_fn=collate_fn,
    )

    if dist.is_initialized():
        dist.barrier(device_ids=[dist.get_rank()])

    return dataloader


def add_new_moe_params(model: Qwen3MoeForCausalLM, ckpt_path: str):
    from safetensors import safe_open

    requested_keys = {"gate_proj", "up_proj", "down_proj"}

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("gate.weight"):
                tensor = f.get_tensor(k)
                name, _t = k.rsplit(".", 1)

                parent = name.removesuffix(".gate").rsplit(".", 1)[0]
                submodule = model.get_submodule(parent)
                submodule.register_module(
                    "gate",
                    torch.nn.Linear(1, 1, bias=False),
                )
                submodule.gate.weight = torch.nn.Parameter(tensor)

            if any(rk in k for rk in requested_keys):
                k: str
                tensor = f.get_tensor(k)
                name, _t = k.rsplit(".", 1)
                name, subname = name.rsplit(".", 1)
                submodule = model.get_submodule(name)
                submodule.register_parameter(subname, torch.nn.Parameter(tensor))

    return model


def get_model_flavour(tcfg: TrainConfig) -> Qwen3MoeForCausalLM:
    if tcfg.model_flavour == "30b":
        config = Qwen3MoeConfig(use_cache=False, dtype=torch.bfloat16)
    elif tcfg.model_flavour == "debug":
        config = Qwen3MoeConfig(
            num_hidden_layers=8,
            use_cache=False,
            num_experts=8,
            dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Model flavour {tcfg.model_flavour} not supported")

    if tcfg.from_old_checkpoint:
        if os.environ.get("USE_NEW_MOE", "false").lower() == "false":
            ckpt_path = (
                "./checkpoints/default-old"
                if tcfg.model_flavour == "debug"
                else "./checkpoints/old-30b"
            )
            model = Qwen3MoeForCausalLM.from_pretrained(ckpt_path)
        else:
            ckpt_path = (
                "./checkpoints/new-debug-full"
                if tcfg.model_flavour == "debug"
                else "./checkpoints/new-30b-full"
            )
            print(f"Loading new MoE params from {ckpt_path}")
            model = Qwen3MoeForCausalLM.from_pretrained(ckpt_path)

    elif tcfg.from_pretrained:
        assert tcfg.model_flavour == "30b", "Can only load 30B from pretrained"
        model = Qwen3MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16
        )
    else:
        model = Qwen3MoeForCausalLM(config)

    if not dist.is_initialized():
        model = model.to("cuda")

    return model

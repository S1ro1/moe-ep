import os
import torch
import sys
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from safetensors.torch import save_file


def main(cfg: AutoConfig, folder_path: str):
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, f) for f in files if f.endswith(".safetensors")]
    _requested_keys = {"gate_proj", "up_proj", "down_proj"}
    new_sd = {}
    for file_path in files:
        print(f"Processing {file_path}")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)
                if not any(rk in k for rk in _requested_keys):
                    new_sd[k] = tensor
                else:
                    parts = k.split(".")
                    expert_id = int(parts[5])
                    param_name = parts[6]

                    new_key = ".".join(parts[:4] + [param_name, "weight"])
                    if new_key not in new_sd:
                        new_sd[new_key] = torch.zeros(
                            cfg.num_experts, tensor.size(0), tensor.size(1)
                        )

                    new_sd[new_key][expert_id] = tensor

    return new_sd


if __name__ == "__main__":
    model_path = sys.argv[1]

    old_ckpt_path = sys.argv[2]
    new_ckpt_path = sys.argv[3]
    skip_model_init = sys.argv[4].lower() if len(sys.argv) > 4 else "false"

    print(f"Loading model from {model_path}")

    os.makedirs(old_ckpt_path, exist_ok=True)
    os.makedirs(new_ckpt_path, exist_ok=True)

    cfg = AutoConfig.from_pretrained(model_path)

    if skip_model_init == "false":
        model = AutoModelForCausalLM.from_config(cfg)
        model.save_pretrained(old_ckpt_path)
        print(f"Saved model to {old_ckpt_path}")
    else:
        print(f"Skipping model init, directly loading from {old_ckpt_path}")

    print(f"Converting checkpoint from {old_ckpt_path} to {new_ckpt_path}")
    new_sd = main(cfg, old_ckpt_path)
    new_ckpt_path = os.path.join(new_ckpt_path, "model.safetensors")
    print(f"Saving new checkpoint to {new_ckpt_path}")

    save_file(new_sd, new_ckpt_path)
    print(f"Copying config.json/generation_config.json to {new_ckpt_path}")
    os.system(f"cp {os.path.join(old_ckpt_path, 'config.json')} {new_ckpt_path.replace('model.safetensors', 'config.json')}")
    os.system(f"cp {os.path.join(old_ckpt_path, 'generation_config.json')} {new_ckpt_path.replace('model.safetensors', 'generation_config.json')}")
    print("Done")

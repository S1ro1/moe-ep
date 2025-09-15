#!/bin/bash
#SBATCH --job-name=30b-ep8-fsdp8-b1-s4096
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --qos=high
#SBATCH --time=12:00:00

export USE_NEW_MOE=true
export EP_SIZE=8

cd /fsx/matej_sirovatka/repros/ep
source /fsx/matej_sirovatka/repros/context-parallel-experiments/.venv/bin/activate

torchrun --nproc-per-node=8 main.py --model-flavour=30b --sequence-length=4096 --run-name=30b-ep8-fsdp8-b1-s4096 --num-steps=20000

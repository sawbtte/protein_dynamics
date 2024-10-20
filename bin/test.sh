#!/bin/bash

#SBATCH -p vip_gpu_gpuuser821  
#SBATCH -o test.out
export PATH=/home/bingxing2/gpuuser834/protein_dynamics

# sleep 20
echo "begin protein dynamics training"

/home/bingxing2/gpuuser834/.conda/envs/proteinDNA/bin/torchrun --nnodes 1 --nproc_per_node=1 --start_method spawn --master_addr 127.0.0.21 --master_port 7447 \
    train.py
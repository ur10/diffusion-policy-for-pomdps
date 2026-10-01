#!/bin/bash

SEEDS=(42)
FREEZE_BLOCKS=(0)

for SEED in "${SEEDS[@]}"; do
    for FREEZE in "${FREEZE_BLOCKS[@]}"; do
        echo "=== Training: algo=videomae env=demo_pusht seed=$SEED freeze_blocks=$FREEZE ==="
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --config_env=configs/envs/robomimic_square.py \
            --config_rl=configs/rl/bc_diffusion_videomae.py \
            --config_rl.wandb_project=diffpo_pomdp_09.05.26 \
            --config_rl.wandb_tag=memdit-compiled-profiler-finetune-${FREEZE} \
            --config_rl.debug=False \
            --config_rl.config_diffusion.finetune_videomae=True \
            --config_rl.config_diffusion.freeze_encoder_blocks=${FREEZE} \
            --seed=${SEED} \
            --batch_size=16 \
            --config_rl.train_steps=150000 \
            --config_rl.eval_steps=5000
    done
done
wait
echo "=== All experiments done ==="

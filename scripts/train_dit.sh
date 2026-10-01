#!/bin/bash

SEEDS=(42)
ENVS=(can square)
OBS_HISTORIES=(1)

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for OBS_HISTORY in "${OBS_HISTORIES[@]}"; do
            echo "=== Training: env=$ENV seed=$SEED obs_history=$OBS_HISTORY ==="
            CUDA_VISIBLE_DEVICES=1 python main.py \
                --config_env=configs/envs/robomimic_${ENV}.py \
                --config_rl=configs/rl/bc_diffusion_dit.py \
                --config_rl.wandb_project=diffpo_pomdp_09.05.26 \
                --config_rl.wandb_tag=bc-diffusion-dit-h-${OBS_HISTORY} \
                --config_rl.debug=False \
                --seed=${SEED} \
                --batch_size=16 \
                --config_rl.config_diffusion.obs_history=${OBS_HISTORY} \
                --config_rl.train_steps=150000 \
                --config_rl.eval_steps=5000
        done
    done
done

wait
echo "=== All experiments done ==="
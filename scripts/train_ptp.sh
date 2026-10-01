#!/bin/bash

SEEDS=(0 42)
ENVS=(can square)
OBS_HISTORIES=(2)

for ENV in "${ENVS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for OBS_HISTORY in "${OBS_HISTORIES[@]}"; do
            echo "=== Launching: env=$ENV seed=$SEED obs_history=$OBS_HISTORY ==="
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --config_env=configs/envs/robomimic_${ENV}.py \
                --config_rl=configs/rl/bc_ptp.py \
                --config_rl.wandb_project=diffpo_pomdp_08.05.26 \
                --config_rl.wandb_tag=ptp_obs_history_${OBS_HISTORY} \
                --config_rl.debug=False \
                --seed=${SEED} \
                --batch_size=16 \
                --config_rl.config_diffusion.obs_history=${OBS_HISTORY} \
                --config_rl.train_steps=100000 \
                --config_rl.eval_steps=5000
        done
    done
done

wait
echo "=== All experiments done ==="
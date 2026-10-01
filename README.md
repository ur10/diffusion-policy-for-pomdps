# Diffusion Policy for POMDPs

Behavioral cloning with diffusion-based policies for partially observable robotic manipulation tasks. Supports observation history conditioning to handle POMDPs, with two diffusion architectures: a UNet-based model and a Diffusion Transformer (DiT).

## Overview

- **Policies**: Diffusion UNet (`bc_diffusion`) and Diffusion Transformer (`bc_diffusion_dit`)
- **Environments**: Robomimic manipulation tasks and PushT-style image-based tasks (`demo_pusht`, `drawer_pusht`, `multi_pusht`)
- **POMDP support**: `obs_history > 1` conditions the policy on a window of past image and proprioceptive frames
- **Action chunking**: predicts `ac_chunk` future actions per forward pass

## Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd diffusion-policy-for-pomdps
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The following packages are **not** in `requirements.txt` and must be installed separately:

**h5py** (for loading HDF5 demo datasets):
```bash
pip install h5py
```

**tqdm** (progress bars during data loading):
```bash
pip install tqdm
```

**robomimic** (simulation environments and dataset utilities — install from source):
```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
cd ..
```

**wandb** (experiment tracking):
```bash
pip install wandb
```

### 3. Download datasets

Download the robomimic demonstration datasets from the [robomimic dataset page](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html). Update the `datadir` field in the relevant env config to point to the downloaded `.hdf5` file:

```python
# configs/envs/robomimic_can.py
config.datadir = "/path/to/datasets/can/ph/low_dim.hdf5"
```

PushT-style tasks use `.zarr` demonstration datasets. Update the `datadir` field in the relevant env config to point to the downloaded dataset:

```python
# configs/envs/drawer_pusht.py
config.datadir = "/path/to/drawer_pusht.zarr"
```

Available PushT env configs:

- `configs/envs/demo_pusht.py`
- `configs/envs/drawer_pusht.py`
- `configs/envs/multi_pusht.py`

## Training

### Past-Token Prediction Policy (PTP)

```bash
python main.py \
    --config_env=configs/envs/robomimic_square.py \
    --config_rl=configs/rl/bc_ptp.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=bc-ptp-square-obs-16 \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64
```

You can override the main long-context settings from the command line:

```bash
python main.py \
    --config_env=configs/envs/robomimic_square.py \
    --config_rl=configs/rl/bc_ptp.py \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=16 \
    --config_rl.config_diffusion.ac_chunk=16 \
    --config_rl.config_diffusion.consistency_samples=4
```

### Diffusion Transformer (DiT)

```bash
python main.py \
    --config_env=configs/envs/robomimic_can.py \
    --config_rl=configs/rl/bc_diffusion_dit.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=bc-diffusion-dit-obs-3 \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=3
```

### Diffusion UNet

```bash
python main.py \
    --config_env=configs/envs/robomimic_can.py \
    --config_rl=configs/rl/bc_diffusion.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=bc-diffusion-unet-obs-3 \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=3
```

### Training On PushT Envs

The PushT envs train through the same `main.py` entrypoint and RL configs as robomimic. The main difference is the env config and the dataset path, which should point to a `.zarr` replay buffer.

#### Demo PushT with DiT

```bash
python main.py \
    --config_env=configs/envs/demo_pusht.py \
    --config_rl=configs/rl/bc_diffusion_dit.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=demo-pusht-dit \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=3
```

#### Drawer PushT with DiT

```bash
python main.py \
    --config_env=configs/envs/drawer_pusht.py \
    --config_rl=configs/rl/bc_diffusion_dit.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=drawer-pusht-dit \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=3
```

#### Multi PushT with PTP

```bash
python main.py \
    --config_env=configs/envs/multi_pusht.py \
    --config_rl=configs/rl/bc_ptp.py \
    --config_rl.wandb_project=diffusion-policy-pomdps \
    --config_rl.wandb_tag=multi-pusht-ptp \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64
```

If you want to run PTP on a PushT env with longer context, use the same overrides as robomimic:

```bash
python main.py \
    --config_env=configs/envs/multi_pusht.py \
    --config_rl=configs/rl/bc_ptp.py \
    --config_rl.debug=False \
    --seed=0 \
    --batch_size=64 \
    --config_rl.config_diffusion.obs_history=16 \
    --config_rl.config_diffusion.ac_chunk=16 \
    --config_rl.config_diffusion.consistency_samples=4
```

## Key configuration options

All options can be overridden from the command line via `--config_rl.<key>=<value>`.

| Flag | Default | Description |
|---|---|---|
| `config_rl.config_diffusion.obs_history` | `1` | Frames of history fed to the policy (`> 1` for POMDP) |
| `config_rl.config_diffusion.ac_chunk` | `16` | Number of future actions to predict |
| `config_rl.config_diffusion.past_action_pred` | `True` in `bc_ptp.py` | Train PTP with past-token prediction loss |
| `config_rl.config_diffusion.past_steps_reg` | `-1` | Restrict PTP past-token loss to the most recent N past steps (`-1` uses all predicted past tokens) |
| `config_rl.config_diffusion.consistency_samples` | `1` | Number of rollout samples used by PTP test-time past-action consistency selection |
| `config_rl.config_diffusion.use_obs` | `False` | Proprioceptive obs strategy: `False`, `"add_token"`, or `"pad_img_tokens"` |
| `config_rl.config_diffusion.encoder.name` | `"resnet"` | Visual encoder: `"resnet"`, `"vit_base"`, `"vit_small"` |
| `config_rl.n_train_demos` | `200` | Number of demonstrations to train on |
| `config_rl.train_steps` | `30000` | Total gradient steps |

from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def bc_diffusion_videomae_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = bc_diffusion_videomae_name_fn
    config.debug = True
    config.wandb_project = "diffusion-policy-pomdps"
    config.wandb_tag = ""

    config.algo = "bc_diffusion_videomae"
    config.profile_training = False

    config.actor_lr = 1e-4

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.n_train_demos = 200
    config.train_steps = 20_000
    config.eval_steps = 1000
    config.action_exec_horizon = 8  # steps to execute per model call (1=temporal ensemble)
    
    # DiffusionTransformerAgent settings
    config.config_diffusion = ConfigDict()

    # HuggingFace VideoMAE model name or local path.
    # Options: "MCG-NJU/videomae-base", "MCG-NJU/videomae-large",
    #          "MCG-NJU/videomae-base-finetuned-kinetics", etc.
    config.config_diffusion.videomae_model = "MCG-NJU/videomae-small-finetuned-kinetics"

    # Set to True to finetune the VideoMAE encoder alongside the denoiser.
    # Uses two optimizer param groups: denoiser at actor_lr, encoder at videomae_lr.
    config.config_diffusion.finetune_videomae = False
    config.config_diffusion.videomae_lr = 1e-5  # 10x lower than actor_lr

    # When finetune_videomae=True, freeze the first N ViT blocks (out of 12 for base)
    # so backprop only runs through the tail. 0 = finetune all blocks.
    # Recommended: 8 (freeze 2/3, train last 4) for ~2x backward-pass speedup.
    config.config_diffusion.freeze_encoder_blocks = 8
    config.config_diffusion.use_videomae_cache = True

    # LoRA finetuning for VideoMAE. Mutually exclusive with finetune_videomae.
    # Freezes all base weights; only trains small low-rank adapter matrices in
    # the Q/K/V projections of every attention layer (~1% of encoder params).
    config.config_diffusion.use_lora = False
    config.config_diffusion.lora_r = 8       # rank — higher = more capacity, more params
    config.config_diffusion.lora_alpha = 16  # scaling factor (alpha/r = effective LR scale)

    # Observation / camera settings.
    config.config_diffusion.n_cams = 1
    config.config_diffusion.use_obs = False

    # VideoMAE always encodes its native frame count (num_frames from model config,
    # typically 16).  obs_history controls how many frames the buffer samples;
    # the encoder pads/truncates to 16 automatically.
    config.config_diffusion.obs_history = 16

    # Action settings
    config.config_diffusion.ac_chunk = 16   # number of actions to predict at once

    # Diffusion schedule.
    config.config_diffusion.train_diffusion_steps = 100
    config.config_diffusion.eval_diffusion_steps = 10

    # DiT denoiser architecture (the only part trained by default).
    # cond_dim is set automatically from encoder.embed_dim (768 for base, 1024 for large).
    config.config_diffusion.noise_net_kwargs = ConfigDict()
    config.config_diffusion.noise_net_kwargs.time_dim = 256
    config.config_diffusion.noise_net_kwargs.hidden_dim = 384
    config.config_diffusion.noise_net_kwargs.num_blocks = 12
    config.config_diffusion.noise_net_kwargs.nhead = 12
    config.config_diffusion.noise_net_kwargs.mlp_ratio = 4.0
    config.config_diffusion.noise_net_kwargs.dropout = 0.1
    # cond_dim and visual_cond_len are set automatically from the encoder at runtime

    return config

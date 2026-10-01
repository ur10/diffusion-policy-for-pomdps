import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def bc_diffusion_jepa_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = bc_diffusion_jepa_name_fn
    config.debug = True
    config.wandb_project = "diffusion-policy-pomdps"
    config.wandb_tag = ""

    config.algo = "bc_diffusion_jepa"

    config.actor_lr = 1e-4

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.n_train_demos = 200
    config.train_steps = 10_000
    config.eval_steps = 500

    config.config_diffusion = ConfigDict()

    # Frozen V-JEPA ViT-L checkpoint (weights are never updated).
    config.config_diffusion.jepa_checkpoint = os.environ.get(
        "JEPA_CHECKPOINT", "models/jepa/checkpoints/vitl16.pth.tar"
    )

    # Set to True to finetune the JEPA encoder end-to-end alongside the denoiser.
    # Uses two optimizer param groups: denoiser at actor_lr, encoder at jepa_lr.
    config.config_diffusion.finetune_jepa = False
    config.config_diffusion.jepa_lr = 1e-5  # 10x lower than actor_lr to preserve pretrained features

    # Observation / camera settings.
    config.config_diffusion.n_cams = 1
    config.config_diffusion.use_obs = False  # False, "add_token", or "pad_img_tokens"

    # Number of past frames fed as a video clip to V-JEPA.
    # Token count per camera = max(obs_history, 2) // 2 * 196
    #   obs_history=1  →  2//2 * 196 =  196 tokens/cam
    #   obs_history=4  →  4//2 * 196 =  392 tokens/cam
    #   obs_history=8  →  8//2 * 196 =  784 tokens/cam
    #   obs_history=16 → 16//2 * 196 = 1568 tokens/cam  (native V-JEPA resolution)
    config.config_diffusion.obs_history = 8

    # Action settings.
    config.config_diffusion.ac_chunk = 16

    # Diffusion schedule.
    config.config_diffusion.train_diffusion_steps = 100
    config.config_diffusion.eval_diffusion_steps = 8

    # DiT denoiser architecture (the only part that is trained).
    config.config_diffusion.noise_net_kwargs = ConfigDict()
    config.config_diffusion.noise_net_kwargs.time_dim = 256
    config.config_diffusion.noise_net_kwargs.hidden_dim = 384
    config.config_diffusion.noise_net_kwargs.num_blocks = 12
    config.config_diffusion.noise_net_kwargs.nhead = 12
    config.config_diffusion.noise_net_kwargs.mlp_ratio = 4.0
    config.config_diffusion.noise_net_kwargs.dropout = 0.1
    config.config_diffusion.noise_net_kwargs.cond_dim = 1024  # JEPA ViT-L embed_dim

    return config

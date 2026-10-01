from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def bc_diffusion_dit_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = bc_diffusion_dit_name_fn
    config.debug = True
    config.wandb_project = "diffusion-policy-pomdps"
    config.wandb_tag = ""

    config.algo = "bc_diffusion_dit"

    config.actor_lr = 1e-4

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.n_train_demos = 100
    config.train_steps = 10_000
    config.eval_steps = 500
    config.action_exec_horizon = 8  # steps to execute per model call (1=temporal ensemble)

    # DiffusionTransformerAgent settings
    config.config_diffusion = ConfigDict()

    # Visual encoder: "resnet", "vit_base", "vit_small"
    config.config_diffusion.encoder = ConfigDict()
    config.config_diffusion.encoder.name = "resnet"
    config.config_diffusion.encoder.size = 18          # ResNet size (18, 34, 50)
    config.config_diffusion.encoder.weights = None
    config.config_diffusion.encoder.restore_path = ""
    config.config_diffusion.encoder.norm_cfg = ConfigDict()
    config.config_diffusion.encoder.norm_cfg.name = "group_norm"
    config.config_diffusion.encoder.norm_cfg.num_groups = 16
    config.config_diffusion.encoder.avg_pool = True    # True=1 token, False=49 tokens

    # Observation / camera settings
    config.config_diffusion.n_cams = 1
    config.config_diffusion.use_obs = False  # False, "add_token", or "pad_img_tokens"

    # Observation history (number of past frames fed as input)
    config.config_diffusion.obs_history = 1  # set > 1 for POMDP history

    # Action settings
    config.config_diffusion.ac_chunk = 16   # number of actions to predict at once

    # Diffusion schedule
    config.config_diffusion.train_diffusion_steps = 100
    config.config_diffusion.eval_diffusion_steps = 10

    # Transformer architecture
    config.config_diffusion.noise_net_kwargs = ConfigDict()
    config.config_diffusion.noise_net_kwargs.time_dim = 256
    config.config_diffusion.noise_net_kwargs.hidden_dim = 512
    config.config_diffusion.noise_net_kwargs.num_blocks = 6
    config.config_diffusion.noise_net_kwargs.dropout = 0.1
    config.config_diffusion.noise_net_kwargs.nhead = 8

    return config

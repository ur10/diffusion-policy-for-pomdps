from typing import Tuple

from ml_collections import ConfigDict

from configs.rl.name_fns import name_fn


def bc_ptp_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = bc_ptp_name_fn
    config.debug = True
    config.wandb_project = "diffusion-policy-pomdps"
    config.wandb_tag = ""

    config.algo = "bc_ptp"
    config.action_exec_horizon = 1  # steps to execute per model call (1=temporal ensemble)

    config.actor_lr = 1e-4
    config.weight_decay = 1e-3
    config.encoder_weight_decay = 1e-6
    config.betas = (0.9, 0.95)

    config.lr_scheduler = "cosine"
    config.lr_warmup_steps = 1000
    config.max_grad_norm = 1.0

    config.use_ema = True
    config.ema_power = 0.75
    config.ema_inv_gamma = 1.0
    config.ema_min_value = 0.0
    config.ema_max_value = 0.9999
    config.ema_update_after_step = 0

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.n_train_demos = 200
    config.train_steps = 30_000
    config.eval_steps = 1000

    config.config_diffusion = ConfigDict()
    config.config_diffusion.transform = "ldp_preproc"

    config.config_diffusion.encoder = ConfigDict()
    config.config_diffusion.encoder.name = "resnet"
    config.config_diffusion.encoder.size = 18
    config.config_diffusion.encoder.weights = None
    config.config_diffusion.encoder.restore_path = ""
    config.config_diffusion.encoder.norm_cfg = ConfigDict()
    config.config_diffusion.encoder.norm_cfg.name = "group_norm"
    config.config_diffusion.encoder.norm_cfg.num_groups = 16
    config.config_diffusion.encoder.avg_pool = True

    config.config_diffusion.n_cams = 1
    # Image observation keys to use as cameras. Must match `n_cams` length.
    # Multi-cam: ["agentview_image", "robot0_eye_in_hand_image"] with n_cams=2.
    config.config_diffusion.cam_keys = ("agentview_image",)
    config.config_diffusion.use_obs = False
    config.config_diffusion.obs_history = 16
    config.config_diffusion.ac_chunk = 16

    config.config_diffusion.train_diffusion_steps = 100
    config.config_diffusion.eval_diffusion_steps = 10

    config.config_diffusion.past_action_pred = True
    config.config_diffusion.past_steps_reg = -1
    config.config_diffusion.consistency_samples = 1

    config.config_diffusion.noise_net_kwargs = ConfigDict()
    config.config_diffusion.noise_net_kwargs.n_layer = 8
    config.config_diffusion.noise_net_kwargs.n_cond_layers = 0
    config.config_diffusion.noise_net_kwargs.n_head = 4
    config.config_diffusion.noise_net_kwargs.n_emb = 256
    config.config_diffusion.noise_net_kwargs.p_drop_emb = 0.0
    config.config_diffusion.noise_net_kwargs.p_drop_attn = 0.3
    config.config_diffusion.noise_net_kwargs.causal_attn = True

    return config

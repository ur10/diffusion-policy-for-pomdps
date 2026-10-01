from configs.rl.name_fns import name_fn
from ml_collections import ConfigDict
from typing import Tuple


def bc_name_fn(config: ConfigDict, *args) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config)
    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = bc_name_fn
    config.debug = True
    config.wandb_project = "diffusion-policy-pomdps"
    config.wandb_tag = ""

    config.algo = "bc"

    config.actor_lr = 3e-4
    
    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)
    config.config_actor.predict_reward = True
    config.config_actor.predict_observation = True

    config.replay_buffer_size = 1e6
    config.replay_buffer_num_episodes = 1e3

    config.datadir = ""    

    return config

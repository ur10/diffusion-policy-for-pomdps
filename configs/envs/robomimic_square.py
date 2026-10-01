import os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from ml_collections import ConfigDict
from typing import Tuple
from configs.envs.terminal_fns import finite_horizon_terminal


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = f"robomimic_{config.env_name}"
    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "robomimic"
    config.env_name = "square"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 10
    config.save_interval = 10
    config.eval_episodes = 5

    config.img_size = 224
    config.datadir = "datasets/square/ph/low_dim_v15.hdf5"

    return config

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
    config.env_name = "tool_hang"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 20
    config.save_interval = 20
    config.eval_episodes = 20

    config.img_size = 256
    config.datadir = ""  # required: path to the robomimic .hdf5 dataset file

    return config

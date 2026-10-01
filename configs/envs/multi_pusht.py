from typing import Tuple

from ml_collections import ConfigDict

from configs.envs.terminal_fns import finite_horizon_terminal


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "pusht"
    config.env_name = "multi_pusht"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 10
    config.save_interval = 10
    config.eval_episodes = 20

    config.img_size = 96
    config.datadir = "/path/to/multi_pusht.zarr"

    return config

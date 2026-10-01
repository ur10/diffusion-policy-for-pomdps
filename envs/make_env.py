import gym
from gym.wrappers import RescaleAction


def make_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    if env_name in {"demo_pusht", "drawer_pusht", "multi_pusht", "demo_pusht_multi"}:
        from envs.pusht.pusht_tasks import TASK_CONFIGS
        import gym.spaces
        import numpy as np
        import types

        spec = TASK_CONFIGS[env_name]
        env = types.SimpleNamespace(
            observation_space=gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(spec["obs_dim"],),
                dtype=np.float32,
            ),
            action_space=gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(spec["ac_dim"],),
                dtype=np.float32,
            ),
            max_episode_steps=spec["max_episode_steps"],
        )
        print("obs space", env.observation_space)
        print("act space", env.action_space)
        return env

    if env_name.startswith("robomimic_"):
        # Return a lightweight handle — obs/act dims come from TASK_CONFIGS.
        # The real env is managed by RoboSuiteTask in the learner.
        from envs.robomimic.robosuite import make_env as _make_robosuite
        from envs.robomimic.robosuite import _MAX_STEPS
        import gym.spaces, types
        task = env_name[len("robomimic_"):]
        from envs.robomimic.robosuite import OBS_KEYS
        import numpy as np

        # set _MAX_STEPS and OBS_KEYS globals without needing the dataset file
        from envs.robomimic import robosuite as _rs
        if task == "transport" and "robot1_eef_pos" not in _rs.OBS_KEYS:
            _rs.OBS_KEYS += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "robot1_gripper_qvel"]
        if task in ("transport", "tool_hang"):
            _rs._MAX_STEPS = 725
        obs_dim = 22 if task == "transport" else 11
        ac_dim  = 14 if task == "transport" else 7

        env = types.SimpleNamespace(
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(ac_dim,), dtype=np.float32),
            max_episode_steps=_MAX_STEPS,
        )
        print("obs space", env.observation_space)
        print("act space", env.action_space)
        return env

    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)

    env.max_episode_steps = getattr(
        env, "max_episode_steps", env.spec.max_episode_steps
    )

    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, -1.0, 1.0)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("obs space", env.observation_space)
    print("act space", env.action_space)

    return env

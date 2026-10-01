# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
from collections import deque

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
try:
    # robosuite < 1.5
    from robosuite.controllers import load_controller_config
    _ROBOSUITE_V15 = False
except ImportError:
    # robosuite >= 1.5 renamed this to load_part_controller_config
    from robosuite.controllers import load_part_controller_config as load_controller_config
    _ROBOSUITE_V15 = True

try:
    from robosuite.controllers import (
        ALL_PART_CONTROLLERS as _ALL_PART,
        ALL_COMPOSITE_CONTROLLERS as _ALL_COMPOSITE,
    )
except ImportError:
    _ALL_PART, _ALL_COMPOSITE = set(), set()
import torch
import tqdm

from buffers.seq_robosuite_buffer import SeqRobosuiteBuffer
from policies.models.dit.transforms import get_transform_by_name, get_gpu_transform_by_name

os.environ["MUJOCO_GL"] = "egl"
OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]
# Some datasets (e.g. the LDP longhist square dump) don't store gripper_qvel.
# `_load_demos_from_obs` overwrites this with the subset actually present so
# both training and eval read the same obs keys.
_ACTIVE_OBS_KEYS = list(OBS_KEYS)
# Default cam set: agentview only. `_configure_for_dataset` may override this
# to include eye-in-hand if the dataset stores it. Override at the config
# level via `config_diffusion.cam_keys` to lock multi-cam on/off.
DEFAULT_CAM_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
_ACTIVE_CAM_KEYS = ["agentview_image"]
_MAX_STEPS = 200
_N_ROLLOUTS = 50
EXP_WEIGHT = 0.0
# Set True by `_load_demos_from_obs` so eval applies LDP's 4-stage reward
# (peg → up → peg → drop) and deterministic reset. Lets a like-for-like
# comparison against the LDP `test/mean_score` metric.
_USE_MULTISTAGE_REWARD = False


def _cam_name_from_key(key):
    """`agentview_image` → `agentview`. `robot0_eye_in_hand_image` → `robot0_eye_in_hand`."""
    return key[: -len("_image")] if key.endswith("_image") else key

BUF_SHUFFLE_RNG = 3904767649

_AC_LOC = None
_AC_SCALE = None


def _normalize_actions(actions):
    loc = _AC_LOC[None].astype(actions.dtype)
    scale = _AC_SCALE[None].astype(actions.dtype)
    return (actions - loc) / scale


def _denormalize_action(actions):
    return actions * _AC_SCALE + _AC_LOC


def load_action_stats(task, dataset_path):
    """Load action normalization stats from dataset without rendering images."""
    global _AC_LOC, _AC_SCALE

    if _AC_LOC is not None and _AC_SCALE is not None:
        return  # Already loaded

    task_path = _get_task_path(task, dataset_path)
    all_actions = []

    with h5py.File(task_path, "r") as f:
        print("Loading action statistics from demonstration data...")
        for demo in f["data"].keys():
            demo_acs = f[f"data/{demo}/actions"][()]
            all_actions.append(demo_acs.astype(np.float32))

    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    _AC_LOC = (max_ac + min_ac) / 2
    _AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)
    print(f"Action stats loaded: loc={_AC_LOC}, scale={_AC_SCALE}")


def _render(env, height=256, width=256, camera_name="agentview"):
    img = env.render(
        mode="rgb_array", height=height, width=width, camera_name=camera_name
    )
    return img.astype(np.uint8)


def _render_cams(env, cam_keys, height=256, width=256):
    """Render every camera in `cam_keys`, returning a list of (H, W, C) uint8 arrays."""
    return [_render(env, height=height, width=width, camera_name=_cam_name_from_key(k))
            for k in cam_keys]


def _obs_dict_to_vec(obs_dict):
    return np.concatenate([obs_dict[k] for k in _ACTIVE_OBS_KEYS]).astype(np.float32)


def _get_task_path(task, dataset_path=None):
    global OBS_KEYS, _MAX_STEPS
    if task == "transport" and "robot1_eef_pos" not in OBS_KEYS:
        OBS_KEYS += [
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
            "robot1_gripper_qvel",
        ]

    if task in ("transport", "tool_hang"):
        _MAX_STEPS = 725

    assert dataset_path is not None, "Must provide dataset_path for robosuite tasks!"
    task_path = os.path.expanduser(dataset_path)
    assert os.path.exists(task_path), f"Missing task data at {task_path}"
    return task_path


def _legacy_to_composite(part_cfg):
    """Wrap a legacy single-part controller config (e.g. {type: OSC_POSE, ...})
    into the composite format robosuite >= 1.5 expects.

    Also performs the legacy → modern field renames:
      damping → damping_ratio
      damping_limits → damping_ratio_limits
      control_delta: bool → input_type: "delta"|"absolute"
    """
    rename_map = {
        "damping": "damping_ratio",
        "damping_limits": "damping_ratio_limits",
    }
    upgraded = {}
    for k, v in part_cfg.items():
        upgraded[rename_map.get(k, k)] = v
    if "control_delta" in upgraded:
        upgraded["input_type"] = "delta" if upgraded.pop("control_delta") else "absolute"
    upgraded.setdefault("gripper", {"type": "GRIP"})
    return {"type": "BASIC", "body_parts": {"right": upgraded}}


def _normalize_controller_config(ctrl_cfg):
    if not isinstance(ctrl_cfg, dict):
        return ctrl_cfg

    ctrl_type = ctrl_cfg.get("type")

    # ---- robosuite >= 1.5: upgrade legacy → composite, pass composite through.
    if _ROBOSUITE_V15:
        if ctrl_type in _ALL_COMPOSITE:
            return ctrl_cfg                                  # already composite
        if ctrl_type in _ALL_PART or ctrl_type is not None:
            return _legacy_to_composite(ctrl_cfg)            # wrap part → composite
        return ctrl_cfg

    # ---- robosuite < 1.5: legacy path (downgrade composite → part if needed).
    if "body_parts" in ctrl_cfg and isinstance(ctrl_cfg["body_parts"], dict):
        body_parts = ctrl_cfg["body_parts"]
        arm_cfg = None
        for key in ("right", "left"):
            if key in body_parts and isinstance(body_parts[key], dict):
                arm_cfg = body_parts[key]
                break
        if arm_cfg is None and len(body_parts) == 1:
            arm_cfg = next(iter(body_parts.values()))
        if isinstance(arm_cfg, dict):
            ctrl_cfg = dict(arm_cfg)

    ctrl_type = ctrl_cfg.get("type")
    if ctrl_type is None:
        return ctrl_cfg

    try:
        default_cfg = load_controller_config(default_controller=ctrl_type)
    except Exception:
        return ctrl_cfg

    # Newer robomimic datasets can store controller kwargs for newer
    # robosuite releases. Fill missing legacy fields from the old
    # robosuite defaults and drop unknown extras that old controllers do
    # not accept.
    normalized = dict(default_cfg)
    rename_map = {
        "damping": "damping_ratio",
        "damping_limits": "damping_ratio_limits",
    }
    for key, value in ctrl_cfg.items():
        key = rename_map.get(key, key)
        if key in default_cfg or key == "type":
            normalized[key] = value
    return normalized


def _normalize_env_meta(env_meta):
    env_meta = dict(env_meta)
    env_kwargs = dict(env_meta.get("env_kwargs", {}))

    # Metadata from newer robosuite / robomimic releases may include
    # kwargs unsupported by the legacy stack used in this repo.
    env_kwargs.pop("lite_physics", None)

    controller_configs = env_kwargs.get("controller_configs")
    if isinstance(controller_configs, list):
        env_kwargs["controller_configs"] = [
            _normalize_controller_config(cfg) for cfg in controller_configs
        ]
    elif controller_configs is not None:
        env_kwargs["controller_configs"] = _normalize_controller_config(
            controller_configs
        )

    env_meta["env_kwargs"] = env_kwargs
    return env_meta


def make_env(task, dataset_path=None):
    task_path = _get_task_path(task, dataset_path)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=task_path)
    env_meta = _normalize_env_meta(env_meta)

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env.reset()
    return env


def _has_states(dataset_path):
    """Detect whether the hdf5 stores per-step sim states (re-render path)."""
    with h5py.File(dataset_path, "r") as f:
        first_demo_key = next(iter(f["data"].keys()))
        return "states" in f[f"data/{first_demo_key}"]


def _configure_for_dataset(dataset_path):
    """Inspect the hdf5 once and set module-level flags consistently for both
    training and eval entry points. Idempotent — safe to call multiple times.

    For obs-only datasets (no `states` key, e.g. LDP longhist) this:
      * narrows `_ACTIVE_OBS_KEYS` to the subset present in obs/
      * narrows `_ACTIVE_CAM_KEYS` to image keys present in obs/
      * bumps `_MAX_STEPS` to 700 (longer episodes than default)
      * enables `_USE_MULTISTAGE_REWARD` so eval applies LDP's 4-stage reward
    """
    global _ACTIVE_OBS_KEYS, _ACTIVE_CAM_KEYS, _MAX_STEPS, _USE_MULTISTAGE_REWARD
    if _has_states(dataset_path):
        return  # default re-render path; module flags already correct

    with h5py.File(dataset_path, "r") as f:
        first = next(iter(f["data"].keys()))
        present = set(f[f"data/{first}/obs"].keys())
        active = [k for k in OBS_KEYS if k in present]
        if not active:
            raise RuntimeError(
                f"Dataset {dataset_path} has no overlap with OBS_KEYS={OBS_KEYS}; "
                f"obs/ contains: {sorted(present)}"
            )
        _ACTIVE_OBS_KEYS = active
        cams = [k for k in DEFAULT_CAM_KEYS if k in present]
        if cams:
            _ACTIVE_CAM_KEYS = cams
        _MAX_STEPS = max(_MAX_STEPS, 700)
        _USE_MULTISTAGE_REWARD = True


def _load_demos_from_obs(task, dataset_path, cam_keys=None):
    """Load demos from a dataset that stores images & lowdim obs directly.

    Used for hdf5 files that lack `states` (e.g. the LDP longhist square dump).
    Mutates `_ACTIVE_OBS_KEYS` to the subset of `OBS_KEYS` actually present in
    the dataset, so eval-time `_obs_dict_to_vec` reads the same keys.

    `cam_keys` is the list of obs keys to load as cameras (e.g.
    ["agentview_image", "robot0_eye_in_hand_image"]). Defaults to the
    module-level `_ACTIVE_CAM_KEYS` if not specified.
    """
    global _AC_LOC, _AC_SCALE, _ACTIVE_OBS_KEYS, _MAX_STEPS, _ACTIVE_CAM_KEYS

    task_path = _get_task_path(task, dataset_path)
    buffer = []
    all_actions = []

    with h5py.File(task_path, "r") as f:
        demos = list(f["data"].keys())
        first_obs_keys = set(f[f"data/{demos[0]}/obs"].keys())
        active = [k for k in OBS_KEYS if k in first_obs_keys]
        if not active:
            raise RuntimeError(
                f"Dataset {dataset_path} has no overlap with OBS_KEYS={OBS_KEYS}; "
                f"obs/ contains: {sorted(first_obs_keys)}"
            )
        _ACTIVE_OBS_KEYS = active

        if cam_keys is None:
            cam_keys = list(_ACTIVE_CAM_KEYS)
        for k in cam_keys:
            if k not in first_obs_keys:
                raise RuntimeError(
                    f"Dataset {dataset_path} is missing image key '{k}'; "
                    f"obs/ contains: {sorted(first_obs_keys)}"
                )
        _ACTIVE_CAM_KEYS = cam_keys

        # Bump rollout cap — these datasets are long-horizon by construction.
        _MAX_STEPS = max(_MAX_STEPS, 700)
        # Enable LDP's 4-stage reward at eval so success numbers are directly
        # comparable to LDP's `test/mean_score`.
        global _USE_MULTISTAGE_REWARD
        _USE_MULTISTAGE_REWARD = True

        print(
            f"Loading demonstration data (obs-based; "
            f"obs_keys={active}; cam_keys={cam_keys})"
        )
        for demo_name in tqdm.tqdm(demos):
            demo = f[f"data/{demo_name}"]
            demo_acs = demo["actions"][()].astype(np.float32)

            images_per_cam = []
            for k in cam_keys:
                imgs = demo[f"obs/{k}"][()]
                if imgs.dtype != np.uint8:
                    imgs = (
                        (np.clip(imgs, 0, 1) * 255).astype(np.uint8)
                        if imgs.max() <= 1.0
                        else imgs.astype(np.uint8)
                    )
                images_per_cam.append(imgs)

            obs_components = [demo[f"obs/{k}"][()].astype(np.float32) for k in active]
            obs_arr = np.concatenate(obs_components, axis=-1)  # (L, sum-of-key-dims)

            buffer.append(
                dict(
                    images=images_per_cam,      # list of n_cams (L, H, W, C) uint8
                    observations=obs_arr,       # (L, obs_dim)
                    actions=demo_acs,           # (L, ac_dim)
                )
            )
            all_actions.append(demo_acs)

    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    _AC_LOC = (max_ac + min_ac) / 2
    _AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)

    for ep in buffer:
        ep["actions"] = _normalize_actions(ep["actions"])

    print("built task", task, "with rollout steps", _MAX_STEPS)
    return buffer


def _load_demos(task, dataset_path, cam_keys=None):
    """Load demos. Auto-routes to the obs-based loader when the dataset lacks
    sim `states` (e.g. the LDP longhist dump). `cam_keys` is the list of obs
    keys to use as cameras (passed through to the obs-loader, applied per-cam
    rendering for the re-render path).
    """
    _configure_for_dataset(dataset_path)
    if cam_keys is None:
        cam_keys = list(_ACTIVE_CAM_KEYS)
    if not _has_states(dataset_path):
        return _load_demos_from_obs(task, dataset_path, cam_keys=cam_keys)

    global _AC_LOC, _AC_SCALE

    render_env = make_env(task, dataset_path)
    task_path = _get_task_path(task, dataset_path)
    buffer = []
    all_actions = []

    with h5py.File(task_path, "r") as f:
        print(f"Loading demonstration data (re-render; cam_keys={cam_keys})")
        for demo in tqdm.tqdm(list(f["data"].keys())):
            sim_states = f[f"data/{demo}/states"][()]
            demo_acs = f[f"data/{demo}/actions"][()]

            images_per_cam = [[] for _ in cam_keys]
            obs, actions = [], []
            for s, a in zip(sim_states, demo_acs):
                obs_dict = render_env.reset_to({"states": s})
                obs.append(_obs_dict_to_vec(obs_dict))
                rendered = _render_cams(render_env, cam_keys)
                for c, im in enumerate(rendered):
                    images_per_cam[c].append(im)
                actions.append(a.astype(np.float32))
                all_actions.append(a.astype(np.float32))

            buffer.append(
                dict(
                    images=[np.array(c) for c in images_per_cam],  # list of (L, H, W, C)
                    observations=np.array(obs),                    # (L, obs_dim)
                    actions=np.array(actions),                     # (L, ac_dim)
                )
            )

    all_actions = np.array(all_actions)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    _AC_LOC = (max_ac + min_ac) / 2
    _AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)

    for ep in buffer:
        ep["actions"] = _normalize_actions(ep["actions"])

    print("built task", task, "with rollout steps", _MAX_STEPS)
    return buffer


class RoboSuiteBuffer(SeqRobosuiteBuffer):
    """
    Loads robosuite demo data into SeqRobosuiteBuffer.
    Samples windows of (obs_history + ac_chunk) steps, returning:
        obs  : (B, obs_history, obs_dim)
        imgs : {"cam{i}": (B, obs_history, C, H, W) for i in range(n_cams)}
        act  : (B, ac_chunk, ac_dim)
        mask : (B, ac_chunk, ac_dim)
    """

    def __init__(
        self,
        task,
        obs_history=1,
        ac_chunk=1,
        n_train_demos=200,
        transform=None,
        dataset_path=None,
        cam_keys=None,
    ):
        if cam_keys is None:
            cam_keys = list(_ACTIVE_CAM_KEYS)
        demos = _load_demos(task, dataset_path, cam_keys=cam_keys)
        assert len(demos) >= n_train_demos, "Not enough demos!"

        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(demos)
        demos = demos[:n_train_demos]

        first = demos[0]
        # `images` is a list of (L, H, W, C) arrays, one per cam
        first_imgs = first["images"]
        if isinstance(first_imgs, np.ndarray):  # back-compat single-cam
            first_imgs = [first_imgs]
        H, W, C = first_imgs[0].shape[1:]
        n_cams = len(first_imgs)
        obs_dim = first["observations"].shape[1]
        ac_dim = first["actions"].shape[1]
        # obs_history-1 prepended frames + L real frames + 1 sentinel per episode
        max_size = sum(len(d["observations"]) + obs_history for d in demos)

        super().__init__(
            max_replay_buffer_size=max_size,
            observation_dim=obs_dim,
            action_dim=ac_dim,
            img_height=H,
            img_width=W,
            img_channels=C,
            obs_history=obs_history,
            ac_chunk=ac_chunk,
            n_cams=n_cams,
        )

        self._transform = transform
        self._cam_keys = cam_keys

        for demo in tqdm.tqdm(demos):
            self.add_episode(
                images=demo["images"],
                observations=demo["observations"],
                actions=demo["actions"],
            )

    def random_episodes(self, batch_size, skip_imgs=False, raw_imgs=False):
        return super().random_episodes(batch_size, transform=self._transform, skip_imgs=skip_imgs, raw_imgs=raw_imgs)

    def get_windows_at_starts(self, starts):
        return super().get_windows_at_starts(starts, transform=self._transform)


# 45-D init state used by LDP's MultimodalSquareWrapper. Indices [10:12]
# are the nut x/y position, which is randomized per episode.
_LDP_LONGHIST_RESET_STATE = np.array(
    [
        0.0, -0.02921895, 0.17810908, 0.02728627, -2.63967499,
        -0.01431297, 2.9502351, 0.77126893, 0.020833, -0.020833,
        -0.11083517, 0.11445349, 0.89, -0.98421243, 0.0,
        0.0, 0.17699121, 10.0, 10.0, 10.0,
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


class _MultiStageSquareWrapper:
    """Eval-time wrapper that mimics LDP's `MultimodalSquareWrapper`:

      * deterministic 45-D init state with randomized nut x/y
      * 4-stage reward: stage advances on (peg → saddle → peg → saddle_drop)
      * `is_success()['task']` only when stage == 4

    Mirrors the logic in
    `ldp/diffusion_policy/env_runner/robomimic_longhist_image_runner.py:45-126`.
    Access pattern differs by one `.env` since the pomdps env is a robomimic
    `EnvRobosuite` directly (LDP wraps it in `RobomimicImageWrapper` first).
    """

    SADDLE_POS = np.array([0.0, 0.0, 0.89 + 0.20])     # ~0.20 m above the peg
    SADDLE_DROP_POS = np.array([0.0, 0.0, 0.89])       # at peg height
    SADDLE_RADIUS = 0.09
    SADDLE_DROP_RADIUS = 0.03
    PEG_ID = 0

    def __init__(self, env):
        self.env = env
        self.rew = 0
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)

    def _raw_robosuite(self):
        """Reach through to the underlying robosuite env."""
        return self.env.env

    def reset(self):
        nut_pos = np.random.uniform([-0.2, -0.2], [0.0, 0.2], size=(2,))
        reset_state = _LDP_LONGHIST_RESET_STATE.copy()
        reset_state[10:12] = nut_pos
        obs = self.env.reset_to({"states": reset_state})
        self.rew = 0
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        rs = self._raw_robosuite()
        nut_world_pos = rs.sim.data.body_xpos[rs.obj_body_id["SquareNut"]]
        on_peg = bool(rs.on_peg(nut_world_pos, self.PEG_ID))

        eef_pos = np.asarray(obs["robot0_eef_pos"])
        on_saddle = np.linalg.norm(eef_pos - self.SADDLE_POS) < self.SADDLE_RADIUS
        on_saddle_drop = (
            np.linalg.norm(eef_pos - self.SADDLE_DROP_POS) < self.SADDLE_DROP_RADIUS
        )

        if   self.rew == 0 and on_peg:        self.rew = 1
        elif self.rew == 1 and on_saddle:     self.rew = 2
        elif self.rew == 2 and on_peg:        self.rew = 3
        elif self.rew == 3 and on_saddle_drop: self.rew = 4

        reward = 1.0 if self.rew == 4 else 0.0
        return obs, reward, done, info

    def is_success(self):
        return {"task": self.rew == 4}

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    # passthroughs for everything else robomimic might call
    def __getattr__(self, name):
        return getattr(self.env, name)


class RoboSuiteTask:
    """Handles eval rollouts against a live robosuite environment."""

    def __init__(
        self,
        task,
        n_cams,
        obs_dim,
        ac_dim,
        obs_history=1,
        consistency_samples=1,
        dataset_path=None,
        eval_transform=None,
        cam_keys=None,
        transform_name="preproc",
        render_size=256,
    ):
        # Make sure module-level flags reflect the dataset format (idempotent
        # — already done by _load_demos during training, but eval skips that).
        if dataset_path is not None:
            _configure_for_dataset(dataset_path)

        if cam_keys is None:
            cam_keys = list(_ACTIVE_CAM_KEYS)
        assert n_cams == len(cam_keys), (
            f"n_cams={n_cams} does not match len(cam_keys)={len(cam_keys)}: "
            f"{cam_keys}"
        )
        # obs_dim depends on which keys the dataset actually carries.
        # Standard `low_dim`/re-render path: 11 (incl. gripper_qvel).
        # LDP-style longhist obs path: 9 (no qvel).
        # Transport: 22 (single-arm) or 14*2 if dual-arm version is added.
        if task == "transport":
            assert obs_dim in (22,) and ac_dim == 14, (
                f"unexpected (obs_dim={obs_dim}, ac_dim={ac_dim}) for transport"
            )
        else:
            assert obs_dim in (9, 11) and ac_dim == 7, (
                f"unexpected (obs_dim={obs_dim}, ac_dim={ac_dim}) for {task}"
            )
        self.eval_env = make_env(task, dataset_path)
        if _USE_MULTISTAGE_REWARD:
            print("[RoboSuiteTask] applying LDP 4-stage reward + deterministic reset")
            self.eval_env = _MultiStageSquareWrapper(self.eval_env)
        self._obs_history = obs_history
        self._consistency_samples = consistency_samples
        self._eval_transform = eval_transform
        self._cam_keys = cam_keys
        self._transform_name = transform_name
        self._render_size = render_size

    def rollout(self, agent, device, n_rollouts=_N_ROLLOUTS, action_exec_horizon=1, n_video_rollouts=0):
        """
        Run n_rollouts with agent.get_actions(imgs, obs).
        obs  passed as (1, obs_history, obs_dim).
        imgs passed as {"cam{i}": (1, obs_history, C, H, W) for i in n_cams}.
        Returns (mean_success, mean_reward, videos) where videos is a list of
        (T, H, W, C) uint8 arrays for the first n_video_rollouts episodes.

        action_exec_horizon: how many env steps to execute from each predicted
        chunk before re-querying the model. >1 skips temporal ensembling but
        reduces model calls by action_exec_horizon×.
        """
        gpu_transform = get_gpu_transform_by_name(self._transform_name).to(device)
        env = self.eval_env
        obs_history = self._obs_history
        cam_keys = self._cam_keys
        n_cams = len(cam_keys)
        success_flags, total_rewards, videos = [], [], []

        def render_and_transform(env_ref):
            raw = _render_cams(env_ref, cam_keys, height=self._render_size, width=self._render_size)
            return (
                [torch.from_numpy(r).permute(2, 0, 1).float() / 255 for r in raw],
                raw,
            )

        def _query_agent(obs_deque, img_deques, executed_action_history):
            obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(device)
            imgs = {
                f"cam{c}": gpu_transform(
                    torch.stack(list(img_deques[c]), dim=0)[None].to(device)
                )
                for c in range(n_cams)
            }
            past_actions = None
            if len(executed_action_history) > 0:
                past_actions = torch.stack(list(executed_action_history), dim=0)[None].to(device)
            with torch.inference_mode():
                if getattr(agent, "supports_consistency_sampling", False):
                    return agent.get_actions(
                        imgs, obs_stacked,
                        past_actions=past_actions,
                        n_samples=self._consistency_samples,
                    )[0].cpu().numpy()
                else:
                    return agent.get_actions(imgs, obs_stacked, obs_valid_len=obs_history)[0].cpu().numpy()

        for i in range(n_rollouts):
            print(f"Rollout {i}", end="\r")
            record = i < n_video_rollouts
            frames = []

            o = env.reset()

            # build initial obs/image
            o_vec = _obs_dict_to_vec(o)
            img_t_list, raw_imgs = render_and_transform(env)  # list of (C, H, W) per cam
            if record:
                frames.append(raw_imgs[0])

            # initialize history by repeating the first real frame — matches the
            # first-frame-repeated warm-up windows written into the training buffer
            obs_deque = deque([torch.from_numpy(o_vec)] * obs_history, maxlen=obs_history)
            img_deques = [
                deque([img_t_list[c]] * obs_history, maxlen=obs_history)
                for c in range(n_cams)
            ]

            done = False
            t = 0
            total_reward = 0
            executed_action_history = deque(maxlen=max(obs_history - 1, 1))

            if action_exec_horizon > 1:
                chunk_step = action_exec_horizon  # trigger re-query on first step
                current_chunk = None
            else:
                act_history = None

            while not done and t < _MAX_STEPS and not env.is_success()["task"]:
                if action_exec_horizon > 1:
                    if chunk_step >= action_exec_horizon:
                        current_chunk = _query_agent(obs_deque, img_deques, executed_action_history)
                        chunk_step = 0
                    ac = _denormalize_action(current_chunk[chunk_step])
                    chunk_step += 1
                else:
                    acs = _query_agent(obs_deque, img_deques, executed_action_history)
                    if act_history is None:
                        act_history = deque(maxlen=len(acs))
                    act_history.append(acs)
                    num_actions = len(act_history)
                    curr_act_preds = np.stack(
                        [
                            pred_actions[idx]
                            for (idx, pred_actions) in zip(
                                range(num_actions - 1, -1, -1), act_history
                            )
                        ]
                    )
                    weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))[::-1]
                    weights = weights / weights.sum()
                    ac = np.sum(weights[:, None] * curr_act_preds, axis=0)
                    ac = _denormalize_action(ac)

                o, r, done, _ = env.step(ac)
                t += 1
                total_reward += r
                executed_action_history.append(torch.from_numpy(ac).float())

                # update history (per cam)
                o_vec = _obs_dict_to_vec(o)
                img_t_list, raw_imgs = render_and_transform(env)
                if record:
                    frames.append(raw_imgs[0])
                obs_deque.append(torch.from_numpy(o_vec))
                for c in range(n_cams):
                    img_deques[c].append(img_t_list[c])

                if done:
                    break

            success_flags.append(float(env.is_success()["task"]))
            total_rewards.append(float(total_reward))
            if record:
                videos.append(np.stack(frames))  # (T, H, W, C)

        return np.mean(success_flags), np.mean(total_rewards), videos

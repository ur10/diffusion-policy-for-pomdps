"""Robomimic HDF5 loader for probe training.

Reads the *same* proprio obs vector + camera images + actions that the
main training pipeline uses (so the encoder sees identical inputs to its
training distribution), but ADDITIONALLY pulls a configurable list of
probe-target obs keys (e.g. SquareNut_pos, SquareNut_quat) and packs them
into a `targets` field on each demo dict.

Does NOT modify `envs.robomimic.robosuite` — only imports helpers from it.
"""

import numpy as np
import h5py
import tqdm

from envs.robomimic import robosuite as rs_mod
from envs.robomimic.robosuite import (
    OBS_KEYS,
    _normalize_actions,
    _get_task_path,
    _configure_for_dataset,
    _has_states,
    _obs_dict_to_vec,
    _render_cams,
    make_env,
    _ACTIVE_CAM_KEYS,
)


# Robomimic exposes a flat per-task `object` vector. For NutAssemblySquare
# the 14-D layout is:
#   [0:3]   SquareNut_to_robot0_eef_pos   (gripper -> nut, world frame)
#   [3:7]   SquareNut_to_robot0_eef_quat  (relative orientation)
#   [7:10]  SquareNut_pos                 (nut position, WORLD frame)
#   [10:14] SquareNut_quat                (nut orientation, world frame)
# We want the world-frame nut pose+quat, so slice 7:14.
DEFAULT_PROBE_KEYS = ("object",)
DEFAULT_TARGET_SLICE = slice(7, 14)


def load_demos_with_targets(
    task,
    dataset_path,
    cam_keys=None,
    probe_keys=DEFAULT_PROBE_KEYS,
    target_slice=DEFAULT_TARGET_SLICE,
):
    """Load demos (obs-mode) and pull both proprio + probe target keys.

    Args:
        probe_keys: HDF5 obs keys to concat into the per-step target vector.
        target_slice: optional slice applied to the concatenated target after
            stacking (e.g. slice(0, 7) extracts pos+quat from the leading
            entries of robomimic's `object` vector). Pass None to keep all dims.

    Returns:
        list of {
            images       : list of n_cams (L, H, W, C) uint8 arrays,
            observations : (L, obs_dim) float32 — proprio (matches training),
            actions      : (L, ac_dim) float32 — normalized via _AC_LOC/_AC_SCALE,
            targets      : (L, target_dim) float32,
        }
    """
    _configure_for_dataset(dataset_path)
    task_path = _get_task_path(task, dataset_path)
    if cam_keys is None:
        cam_keys = list(_ACTIVE_CAM_KEYS)

    if _has_states(dataset_path):
        return _load_via_state_replay(
            task, task_path, cam_keys, probe_keys, target_slice
        )

    buffer = []
    all_actions = []

    with h5py.File(task_path, "r") as f:
        demos = list(f["data"].keys())
        first_obs_keys = set(f[f"data/{demos[0]}/obs"].keys())

        active_obs = [k for k in OBS_KEYS if k in first_obs_keys]
        if not active_obs:
            raise RuntimeError(
                f"Dataset {dataset_path} has no overlap with OBS_KEYS={OBS_KEYS}; "
                f"obs/ contains: {sorted(first_obs_keys)}"
            )

        missing_probe = [k for k in probe_keys if k not in first_obs_keys]
        if missing_probe:
            raise RuntimeError(
                f"Dataset {dataset_path} is missing probe keys {missing_probe}. "
                f"Available obs keys: {sorted(first_obs_keys)}"
            )

        missing_cam = [k for k in cam_keys if k not in first_obs_keys]
        if missing_cam:
            raise RuntimeError(
                f"Dataset {dataset_path} is missing image keys {missing_cam}. "
                f"Available obs keys: {sorted(first_obs_keys)}"
            )

        print(
            f"Loading probe demos "
            f"(obs_keys={active_obs}, cam_keys={cam_keys}, probe_keys={list(probe_keys)})"
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

            obs_components = [
                demo[f"obs/{k}"][()].astype(np.float32) for k in active_obs
            ]
            obs_arr = np.concatenate(obs_components, axis=-1)

            target_components = [
                demo[f"obs/{k}"][()].astype(np.float32) for k in probe_keys
            ]
            target_arr = np.concatenate(target_components, axis=-1)
            if target_slice is not None:
                target_arr = target_arr[:, target_slice]

            buffer.append(
                dict(
                    images=images_per_cam,
                    observations=obs_arr,
                    actions=demo_acs,
                    targets=target_arr,
                )
            )
            all_actions.append(demo_acs)

    # match main loader: compute action normalization from this dataset and
    # set module-level _AC_LOC / _AC_SCALE so eval-time helpers stay consistent
    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    rs_mod._AC_LOC = (max_ac + min_ac) / 2
    rs_mod._AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)

    for ep in buffer:
        ep["actions"] = _normalize_actions(ep["actions"])

    return buffer


def _load_via_state_replay(task, task_path, cam_keys, probe_keys, target_slice):
    """Mirror of the state-replay branch in `envs.robomimic.robosuite._load_demos`,
    plus per-step probe-target extraction from the same obs_dict that
    `reset_to(...)` returns. Used when the HDF5 has `states` but not `obs/*`.
    """
    render_env = make_env(task, task_path)
    buffer = []
    all_actions = []

    with h5py.File(task_path, "r") as f:
        # Validate probe keys against the first state's obs dict so we fail
        # fast rather than after a full demo render.
        sample_state = f[f"data/{next(iter(f['data']))}/states"][0]
        sample_obs = render_env.reset_to({"states": sample_state})
        missing = [k for k in probe_keys if k not in sample_obs]
        if missing:
            raise RuntimeError(
                f"Re-render env for task {task!r} missing probe keys {missing}. "
                f"obs_dict has: {sorted(sample_obs.keys())}"
            )
        for k in cam_keys:
            if k not in sample_obs and not k.endswith("_image"):
                raise RuntimeError(
                    f"cam key {k!r} not produced by env; obs_dict has "
                    f"{sorted(sample_obs.keys())}"
                )

        print(
            f"Loading probe demos (re-render; "
            f"cam_keys={cam_keys}, probe_keys={list(probe_keys)})"
        )
        for demo in tqdm.tqdm(list(f["data"].keys())):
            sim_states = f[f"data/{demo}/states"][()]
            demo_acs = f[f"data/{demo}/actions"][()].astype(np.float32)

            images_per_cam = [[] for _ in cam_keys]
            obs_list, target_list = [], []
            for s in sim_states:
                obs_dict = render_env.reset_to({"states": s})
                obs_list.append(_obs_dict_to_vec(obs_dict))
                rendered = _render_cams(render_env, cam_keys)
                for c, im in enumerate(rendered):
                    images_per_cam[c].append(im)
                target_list.append(
                    np.concatenate(
                        [obs_dict[k].astype(np.float32) for k in probe_keys], axis=-1
                    )
                )

            target_arr = np.stack(target_list, axis=0)
            if target_slice is not None:
                target_arr = target_arr[:, target_slice]

            buffer.append(
                dict(
                    images=[np.array(c) for c in images_per_cam],
                    observations=np.array(obs_list, dtype=np.float32),
                    actions=demo_acs,
                    targets=target_arr,
                )
            )
            all_actions.append(demo_acs)

    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    rs_mod._AC_LOC = (max_ac + min_ac) / 2
    rs_mod._AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)

    for ep in buffer:
        ep["actions"] = _normalize_actions(ep["actions"])

    return buffer

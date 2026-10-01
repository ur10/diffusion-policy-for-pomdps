"""Eval a trained probe MLP by overlaying its predicted object pose as a
cross marker on rollout video frames.

Per rollout:
  * Build a fresh robosuite env via `make_env(task, dataset_path)`.
  * Each step:
      - DiT policy chooses action (same temporal-ensemble logic as the
        original training-time rollout).
      - Encoder rep -> probe MLP -> predicted pos+quat (frozen).
      - Render the eval camera, project predicted xyz to pixel coords,
        draw a RED cross. Optionally draw GREEN cross for ground truth.
  * Save the per-rollout frames as an mp4.

Use the probe ckpt produced by `train_probe.py --save_probe <path>`.

Example:
    python -m probe.eval_probe \\
        --ckpt logs/robomimic/can/.../save/agent_045000_perf1.000.pt \\
        --probe_ckpt probe_can.pt \\
        --dataset_path datasets/can/ph/low_dim_v15.hdf5 \\
        --task can \\
        --img_size 256 --obs_history 1 --ac_chunk 16 --n_cams 1 \\
        --hidden_dim 512 --num_blocks 6 --nhead 8 \\
        --transform_name preproc \\
        --n_rollouts 5 --max_steps 400
"""

import argparse
import os
from collections import deque

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm
from ml_collections import ConfigDict

from policies.models.bc_policy_dit import BCPolicy_DiT
from policies.models.dit.transforms import (
    get_transform_by_name,
    get_gpu_transform_by_name,
)
from envs.robomimic import robosuite as rs_mod
from envs.robomimic.robosuite import (
    make_env,
    _render_cams,
    _obs_dict_to_vec,
    _configure_for_dataset,
)
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)

from train_probe import ProbeMLP


# --------------------------------------------------------------------------- #
# Camera projection + drawing
# --------------------------------------------------------------------------- #
def _get_sim(env):
    """Robomimic wraps robosuite; the MuJoCo sim sits one level deeper."""
    for path in ("sim", "env.sim", "base_env.sim"):
        obj = env
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find MuJoCo sim object on env")


def world_to_pixel(sim, point_xyz, camera_name, H, W):
    K = get_camera_intrinsic_matrix(sim, camera_name, H, W)
    T = get_camera_extrinsic_matrix(sim, camera_name)
    p_world = np.append(point_xyz, 1.0)
    p_cam = T @ p_world
    if p_cam[2] <= 0:
        return None
    p_pix = K @ p_cam[:3]
    u, v = p_pix[0] / p_pix[2], p_pix[1] / p_pix[2]
    # v = H - 1 - v   # robomimic returns top-left origin; MuJoCo intrinsic is bottom-left
    return int(round(u)), int(round(v))


def draw_cross(img, uv, color, size=18, thickness=2):
    if uv is None:
        return img
    h, w = img.shape[:2]
    if not (0 <= uv[0] < w and 0 <= uv[1] < h):
        return img
    cv2.drawMarker(
        img, uv, color=color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size, thickness=thickness, line_type=cv2.LINE_AA,
    )
    return img


# --------------------------------------------------------------------------- #
# Policy config builder (mirrors train_probe.py / probe_dit_encoder.py)
# --------------------------------------------------------------------------- #
def build_config_rl(args):
    cfg = ConfigDict()
    cfg.actor_lr = 1e-4
    cd = ConfigDict()
    cd.encoder = ConfigDict()
    cd.encoder.name = args.encoder_name
    cd.encoder.size = args.encoder_size
    cd.encoder.weights = None
    cd.encoder.restore_path = ""
    cd.encoder.norm_cfg = ConfigDict()
    cd.encoder.norm_cfg.name = "group_norm"
    cd.encoder.norm_cfg.num_groups = args.norm_groups
    cd.encoder.avg_pool = args.avg_pool
    cd.n_cams = args.n_cams
    cd.use_obs = args.use_obs
    cd.obs_history = args.obs_history
    cd.ac_chunk = args.ac_chunk
    cd.train_diffusion_steps = args.train_diffusion_steps
    cd.eval_diffusion_steps = args.eval_diffusion_steps

    nn_kw = ConfigDict()
    nn_kw.time_dim = args.time_dim
    nn_kw.hidden_dim = args.hidden_dim
    nn_kw.num_blocks = args.num_blocks
    nn_kw.dropout = args.dropout
    nn_kw.nhead = args.nhead
    nn_kw.dim_feedforward = args.dim_feedforward
    nn_kw.activation = "gelu"
    nn_kw.causal_encoder = False
    cd.noise_net_kwargs = nn_kw
    cfg.config_diffusion = cd
    return cfg


# --------------------------------------------------------------------------- #
# Action stats + dims pulled directly from the dataset
# --------------------------------------------------------------------------- #
def init_action_stats_and_dims(dataset_path):
    """Compute _AC_LOC/_AC_SCALE and derive (obs_dim, ac_dim) from the HDF5."""
    _configure_for_dataset(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        all_actions = np.concatenate(
            [f[f"data/{d}/actions"][()].astype(np.float32) for d in demos], axis=0
        )
        first = demos[0]

        # obs_dim: sum of dims of the active OBS_KEYS subset for this dataset.
        # For state-only datasets we fall back to OBS_KEYS / 9-D, which the
        # caller can override via --obs_dim if needed.
        present = set(f[f"data/{first}/obs"].keys()) if "obs" in f[f"data/{first}"] else set()
        if present:
            obs_dim = sum(
                f[f"data/{first}/obs/{k}"].shape[1] for k in rs_mod._ACTIVE_OBS_KEYS
            )
        else:
            obs_dim = None   # caller must supply

        ac_dim = f[f"data/{first}/actions"].shape[1]

    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    rs_mod._AC_LOC = (max_ac + min_ac) / 2
    rs_mod._AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)
    return obs_dim, ac_dim


def _denormalize_action(a):
    return a * rs_mod._AC_SCALE + rs_mod._AC_LOC


# --------------------------------------------------------------------------- #
# Per-rollout loop
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_rollout(
    rollout_idx, agent, probe, args, gpu_transform, cpu_transform, output_dir,
):
    env = make_env(args.task, args.dataset_path)
    sim = _get_sim(env)
    o = env.reset()

    cam_keys = args.cam_keys
    n_cams = args.n_cams
    cam_name = cam_keys[0].replace("_image", "")
    H = W = args.img_size

    # Build initial obs/img deques (warm-up by repeating first frame, mirroring
    # the training buffer's add_episode padding).
    o_vec = _obs_dict_to_vec(o)
    raw0 = _render_cams(env, cam_keys, height=H, width=W)
    img_t0 = [torch.from_numpy(r).permute(2, 0, 1).float() / 255 for r in raw0]

    obs_deque = deque(
        [torch.from_numpy(o_vec)] * args.obs_history, maxlen=args.obs_history
    )
    img_deques = [
        deque([img_t0[c]] * args.obs_history, maxlen=args.obs_history)
        for c in range(n_cams)
    ]
    executed_action_history = deque(maxlen=max(args.obs_history - 1, 1))

    frames = []
    pos_errs = []
    quat_errs = []
    act_history = None
    chunk_step = args.action_exec_horizon
    current_chunk = None

    t = 0
    done = False
    success = False
    EXP_WEIGHT = 0.0

    while not done and t < args.max_steps and not env.is_success()["task"]:
        # ---- build encoder input
        obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(args.device)
        imgs_stacked = {
            f"cam{c}": torch.stack(list(img_deques[c]), dim=0)[None].to(args.device)
            for c in range(n_cams)
        }
        if gpu_transform is not None:
            imgs_stacked = {k: gpu_transform(v) for k, v in imgs_stacked.items()}
        elif cpu_transform is not None:
            for k, v in imgs_stacked.items():
                B, T, C, Hi, Wi = v.shape
                vt = cpu_transform(v.reshape(B * T, C, Hi, Wi))
                imgs_stacked[k] = vt.reshape(B, T, *vt.shape[1:])

        # ---- probe forward
        s_t = agent.tokenize_obs(imgs_stacked, obs_stacked)
        enc_cache = agent.noise_net.forward_enc(s_t)
        rep = enc_cache[-1].permute(1, 0, 2).reshape(1, -1)
        pred = probe(rep).cpu().numpy()[0]
        pred_xyz = pred[:3]

        # ---- policy action (same logic as RoboSuiteTask.rollout)
        if args.action_exec_horizon > 1:
            if chunk_step >= args.action_exec_horizon:
                current_chunk = agent.get_actions(
                    imgs_stacked, obs_stacked, obs_valid_len=args.obs_history
                )[0].cpu().numpy()
                chunk_step = 0
            ac = _denormalize_action(current_chunk[chunk_step])
            chunk_step += 1
        else:
            acs = agent.get_actions(
                imgs_stacked, obs_stacked, obs_valid_len=args.obs_history
            )[0].cpu().numpy()
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

        # ---- render frame + draw markers
        frame = _render_cams(env, [cam_keys[0]], height=H, width=W)[0].copy()
        pred_uv = world_to_pixel(sim, pred_xyz, cam_name, H, W)
        draw_cross(frame, pred_uv, color=(255, 0, 0))  # red = prediction

        gt_xyz = None
        gt_sliced = None
        if args.gt_obs_key and args.gt_obs_key in o:
            gt_full = np.asarray(o[args.gt_obs_key], dtype=np.float32)
            gt_sliced = gt_full[args.gt_slice_obj] if args.gt_slice_obj is not None else gt_full
            gt_xyz = gt_sliced[:3]
            gt_uv = world_to_pixel(sim, gt_xyz, cam_name, H, W)
            draw_cross(frame, gt_uv, color=(0, 255, 0))  # green = ground truth

        # On-frame text: step + per-step pos error
        if gt_xyz is not None:
            pos_err = float(np.linalg.norm(pred_xyz - gt_xyz))
            pos_errs.append(pos_err)
            if pred.shape[0] >= 7 and gt_sliced.shape[0] >= 7:
                gt_quat = gt_sliced[3:7]
                quat_errs.append(float(np.linalg.norm(pred[3:7] - gt_quat)))
            cv2.putText(
                frame, f"t={t} pos_err={pos_err:.3f}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        frames.append(frame)

        # ---- step env
        o, r, done, _ = env.step(ac)
        t += 1
        executed_action_history.append(torch.from_numpy(ac).float())

        # update history
        o_vec = _obs_dict_to_vec(o)
        new_raw = _render_cams(env, cam_keys, height=H, width=W)
        new_img_t = [torch.from_numpy(r).permute(2, 0, 1).float() / 255 for r in new_raw]
        obs_deque.append(torch.from_numpy(o_vec))
        for c in range(n_cams):
            img_deques[c].append(new_img_t[c])

    success = bool(env.is_success()["task"])
    tag = "success" if success else "fail"
    video_path = os.path.join(
        output_dir, f"rollout_{rollout_idx:03d}_{tag}_steps{t}.mp4"
    )
    imageio.mimwrite(video_path, np.stack(frames), fps=args.fps, quality=8)
    mean_pos = float(np.mean(pos_errs)) if pos_errs else float("nan")
    mean_quat = float(np.mean(quat_errs)) if quat_errs else float("nan")
    print(
        f"rollout {rollout_idx}: steps={t} {tag} "
        f"mean_pos_err={mean_pos:.4f} mean_quat_err={mean_quat:.4f} "
        f"-> {video_path}"
    )
    return success, mean_pos


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="DiT agent_*.pt")
    p.add_argument("--probe_ckpt", required=True,
                   help="probe ckpt saved by train_probe.py --save_probe")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--task", default="can")
    p.add_argument("--cam_keys", nargs="+", default=["agentview_image"])
    p.add_argument(
        "--gt_obs_key", default="object",
        help="env obs_dict key for ground-truth pose. The --target_slice is "
             "applied first; then dims [:3] are drawn as the green cross. "
             "Set to '' to skip GT drawing.",
    )
    p.add_argument(
        "--target_slice", default="7:14",
        help='Python slice applied to obs_dict[gt_obs_key] before extracting '
             'pos/quat. Must match the slice used in train_probe.py so the '
             'green cross corresponds to what the probe was trained to predict. '
             'For square `object`, 7:14 = world-frame nut pose+quat.',
    )

    # policy hyperparams (match training)
    p.add_argument("--img_size", type=int, default=256, help="render H=W")
    p.add_argument("--obs_history", type=int, default=1)
    p.add_argument("--ac_chunk", type=int, default=16)
    p.add_argument("--n_cams", type=int, default=1)
    p.add_argument("--use_obs", default=False)
    p.add_argument("--encoder_name", default="resnet")
    p.add_argument("--encoder_size", type=int, default=18)
    p.add_argument("--norm_groups", type=int, default=16)
    p.add_argument("--avg_pool", action="store_true", default=True)
    p.add_argument("--no_avg_pool", dest="avg_pool", action="store_false")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--train_diffusion_steps", type=int, default=100)
    p.add_argument("--eval_diffusion_steps", type=int, default=8)
    p.add_argument("--transform_name", default="preproc")
    p.add_argument("--action_exec_horizon", type=int, default=1)

    # eval / video
    p.add_argument("--n_rollouts", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--output_dir", default="probe_eval_videos")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    use_obs = args.use_obs
    if isinstance(use_obs, str) and use_obs.lower() in ("false", "0", "none", ""):
        use_obs = False
    args.use_obs = use_obs

    # Parse --target_slice ("7:14" / "" / "none")
    if args.target_slice and args.target_slice.lower() not in ("none",):
        parts = [p.strip() for p in args.target_slice.split(":")]
        args.gt_slice_obj = slice(*[int(p) if p else None for p in parts])
    else:
        args.gt_slice_obj = None

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------- dataset stats / dims
    obs_dim, ac_dim = init_action_stats_and_dims(args.dataset_path)
    if obs_dim is None:
        raise RuntimeError(
            "Could not derive obs_dim from dataset (no obs/ group). "
            "Pass --obs_dim explicitly if you support that flag, or use a "
            "dataset that exposes proprio under data/<demo>/obs/*."
        )
    print(f"obs_dim={obs_dim} ac_dim={ac_dim}")

    # -------------------------------------------------------------- policy
    config_rl = build_config_rl(args)
    print("Building BCPolicy_DiT ...")
    policy = BCPolicy_DiT(
        obs_dim=obs_dim,
        action_dim=ac_dim,
        config_seq=None,
        config_rl=config_rl,
    ).to(args.device)

    sd = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"  ckpt loaded ({len(missing)} missing, {len(unexpected)} unexpected)")

    agent = policy.agent
    agent.eval()
    for p_ in agent.parameters():
        p_.requires_grad_(False)

    # --------------------------------------------------------------- probe
    probe_data = torch.load(args.probe_ckpt, map_location=args.device, weights_only=True)
    in_dim = probe_data["in_dim"]
    probe_hidden_dim = probe_data["probe_hidden_dim"]
    target_dim = probe_data["target_dim"]
    probe = ProbeMLP(in_dim, probe_hidden_dim, target_dim).to(args.device)
    probe.load_state_dict(probe_data["probe_state_dict"])
    probe.eval()                       # <-- probe in eval mode
    for p_ in probe.parameters():
        p_.requires_grad_(False)
    print(
        f"probe loaded: in_dim={in_dim} hidden={probe_hidden_dim} "
        f"target_dim={target_dim}"
    )

    # ----------------------------------------------------------- transform
    try:
        gpu_transform = get_gpu_transform_by_name(args.transform_name).to(args.device)
        cpu_transform = None
    except NotImplementedError:
        gpu_transform = None
        cpu_transform = get_transform_by_name(args.transform_name)
    print(f"transform: {'GPU' if gpu_transform is not None else 'CPU'} "
          f"({args.transform_name})")

    # ----------------------------------------------------------- rollouts
    successes, mean_pos = [], []
    for i in tqdm.trange(args.n_rollouts, desc="rollouts"):
        s, p_err = run_rollout(
            i, agent, probe, args, gpu_transform, cpu_transform, args.output_dir,
        )
        successes.append(int(s))
        if not np.isnan(p_err):
            mean_pos.append(p_err)

    print()
    print(f"success rate: {np.mean(successes) * 100:.1f}% "
          f"({sum(successes)}/{len(successes)})")
    if mean_pos:
        print(f"avg per-step pos error: {np.mean(mean_pos):.4f} m")
    print(f"videos in {args.output_dir}")


if __name__ == "__main__":
    main()

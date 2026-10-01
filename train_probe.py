"""Train a 4-layer MLP probe on a frozen DiT encoder.

Probe protocol:
  * Build BCPolicy_DiT with config matching the trained checkpoint.
  * Load checkpoint weights into the LIVE agent (the checkpoint does not
    contain the EMA shadow), freeze everything, set eval mode.
  * For each obs_history window: run agent.tokenize_obs -> noise_net.forward_enc,
    take enc_cache[-1] of shape (n_tokens, B, hidden_dim), permute and flatten
    to (B, n_tokens * hidden_dim).
  * Feed that into a 4-layer MLP -> (B, target_dim).
  * Target = nut pos+quat (7 dims) at the current/last frame of the window.
  * MSE loss; Adam on probe params only.

This script is fully separate from the policy training pipeline — it touches
no shared infrastructure. Toggle / configure entirely through the CLI flags
below; in particular, `--probe_keys` is the only training-data assumption
beyond what the policy checkpoint already implies.

Example:
    python train_probe.py \\
        --ckpt logs/robomimic/square/<run>/save/agent_XXXXX.pt \\
        --dataset_path /path/to/robomimic_square.hdf5 \\
        --task square \\
        --obs_history 3 --ac_chunk 16 --n_cams 1 \\
        --hidden_dim 512 --num_blocks 6 --nhead 8 \\
        --img_size 84 --obs_dim 23 --ac_dim 7 \\
        --train_demos 180 --val_demos 20 \\
        --steps 5000 --batch_size 64
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ml_collections import ConfigDict

from policies.models.bc_policy_dit import BCPolicy_DiT
from policies.models.dit.transforms import (
    get_transform_by_name,
    get_gpu_transform_by_name,
)
from probe.probe_buffer import ProbeBuffer
from probe.probe_loader import load_demos_with_targets


BUF_SHUFFLE_RNG = 1234   # matches `BUF_SHUFFLE_RNG` in envs/robomimic/robosuite.py


# --------------------------------------------------------------------------- #
# Policy config builder (mirrors probe_dit_encoder.py)
# --------------------------------------------------------------------------- #
def build_config_rl(args):
    cfg = ConfigDict()
    cfg.actor_lr = 1e-4   # unused at inference, but BCPolicy_DiT reads it

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
# Buffer construction (train/val split on disjoint demo subsets)
# --------------------------------------------------------------------------- #
def build_buffers(demos, train_n, val_n, obs_history, ac_chunk):
    rng = random.Random(BUF_SHUFFLE_RNG)
    rng.shuffle(demos)
    assert train_n + val_n <= len(demos), (
        f"Asked for {train_n}+{val_n} demos but only {len(demos)} available"
    )
    train_demos = demos[:train_n]
    val_demos = demos[train_n:train_n + val_n]

    first = train_demos[0]
    H, W, C = first["images"][0].shape[1:]
    n_cams = len(first["images"])
    obs_dim = first["observations"].shape[1]
    ac_dim = first["actions"].shape[1]
    target_dim = first["targets"].shape[1]

    def _make(demo_list):
        max_size = sum(len(d["observations"]) + obs_history for d in demo_list)
        buf = ProbeBuffer(
            max_replay_buffer_size=max_size,
            observation_dim=obs_dim,
            action_dim=ac_dim,
            img_height=H,
            img_width=W,
            img_channels=C,
            obs_history=obs_history,
            ac_chunk=ac_chunk,
            n_cams=n_cams,
            target_dim=target_dim,
        )
        for d in tqdm.tqdm(demo_list, desc="filling buffer"):
            buf.add_episode(
                images=d["images"],
                observations=d["observations"],
                actions=d["actions"],
                targets=d["targets"],
            )
        return buf

    return _make(train_demos), _make(val_demos)


# --------------------------------------------------------------------------- #
# Probe MLP — 4 linear layers, GELU, MSE-trained
# --------------------------------------------------------------------------- #
class ProbeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Encoder forward: imgs/obs -> flattened final-layer encoder rep
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode(agent, batch, gpu_transform, device):
    obs = batch["obs"].to(device, non_blocking=True)
    imgs = {}
    for cam, v in batch["imgs"].items():
        v = v.to(device, non_blocking=True)
        if gpu_transform is not None:
            v = v.float().div_(255.0)
            v = gpu_transform(v)   # transforms accept (B, T, C, H, W) directly
        imgs[cam] = v
    s_t = agent.tokenize_obs(imgs, obs)                # (B, n_tokens, token_dim)
    enc_cache = agent.noise_net.forward_enc(s_t)       # list of (n_tokens, B, hidden_dim)
    rep = enc_cache[-1]                                # (n_tokens, B, hidden_dim)
    B = rep.shape[1]
    rep = rep.permute(1, 0, 2).reshape(B, -1)          # (B, n_tokens * hidden_dim)
    return rep


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    # Checkpoint + data
    p.add_argument("--ckpt", required=True, help="path to agent_*.pt")
    p.add_argument("--dataset_path", required=True, help="robomimic HDF5 path")
    p.add_argument("--task", default="square")
    p.add_argument("--cam_keys", nargs="+", default=["agentview_image"])
    p.add_argument(
        "--probe_keys", nargs="+", default=["object"],
        help="HDF5 obs keys to concat as the probe target. Default is "
             "robomimic's flat `object` vector (14-D for square).",
    )
    p.add_argument(
        "--target_slice", default="7:14",
        help='Python slice spec applied to the concatenated target. For square '
             'the 14-D `object` vector has world-frame nut pose+quat at 7:14 '
             '(0:7 is the gripper-relative vector, not what we want). Pass '
             '"none" or "" to disable.',
    )

    # Image size (must match training); obs_dim/ac_dim are auto-derived from
    # the first loaded demo so the proprio dim never has to be guessed.
    p.add_argument("--img_size", type=int, default=84)

    # Policy hyperparams (must match training)
    p.add_argument("--obs_history", type=int, default=3)
    p.add_argument("--ac_chunk", type=int, default=16)
    p.add_argument("--n_cams", type=int, default=1)
    p.add_argument("--use_obs", default=False,
                   help='False, "add_token", or "pad_img_tokens"')
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

    # Probe training
    p.add_argument("--probe_hidden_dim", type=int, default=1024)
    p.add_argument("--train_demos", type=int, default=180)
    p.add_argument("--val_demos", type=int, default=20)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_probe",
                   default="probe.pt", help="path to save the trained probe")

    args = p.parse_args()

    # Coerce string-False from CLI
    use_obs = args.use_obs
    if isinstance(use_obs, str) and use_obs.lower() in ("false", "0", "none", ""):
        use_obs = False
    args.use_obs = use_obs

    # Parse --target_slice ("0:7" / "" / "none")
    if args.target_slice and args.target_slice.lower() not in ("none",):
        parts = [p.strip() for p in args.target_slice.split(":")]
        target_slice = slice(*[int(p) if p else None for p in parts])
    else:
        target_slice = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------- load demos
    demos = load_demos_with_targets(
        task=args.task,
        dataset_path=args.dataset_path,
        cam_keys=args.cam_keys,
        probe_keys=tuple(args.probe_keys),
        target_slice=target_slice,
    )
    first = demos[0]
    obs_dim = first["observations"].shape[1]
    ac_dim = first["actions"].shape[1]
    target_dim = first["targets"].shape[1]
    print(f"loaded {len(demos)} demos | "
          f"obs_dim={obs_dim} ac_dim={ac_dim} target_dim={target_dim}")

    # ---------------------------------------------------------------- policy
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
    print(f"  loaded {args.ckpt}")
    print(f"  missing  ({len(missing)}): "
          f"{missing[:2]}{' ...' if len(missing) > 2 else ''}")
    print(f"  unexpect ({len(unexpected)}): {unexpected[:2]}")
    print("NOTE: probing the LIVE trained agent (the checkpoint does not "
          "contain EMA shadow weights for this run).")

    # Freeze everything, use live agent in eval mode (BN/dropout off)
    agent = policy.agent
    agent.eval()
    for p_ in agent.parameters():
        p_.requires_grad_(False)

    # -------------------------------------------------------------- transform
    cpu_transform = None
    gpu_transform = None
    try:
        gpu_transform = get_gpu_transform_by_name(args.transform_name).to(args.device)
    except NotImplementedError:
        cpu_transform = get_transform_by_name(args.transform_name)
    use_gpu_transform = gpu_transform is not None
    print(f"transform: {'GPU' if use_gpu_transform else 'CPU'} ({args.transform_name})")

    # ---------------------------------------------------------------- buffers
    train_buf, val_buf = build_buffers(
        demos, args.train_demos, args.val_demos, args.obs_history, args.ac_chunk
    )
    print(f"train buffer size: {len(train_buf)}, val buffer size: {len(val_buf)}")

    # ------------------------------------------------------------------ probe
    # Discover rep dimension via a dummy forward
    sample_kwargs = dict(
        transform=None if use_gpu_transform else cpu_transform,
        raw_imgs=use_gpu_transform,
    )
    dummy = train_buf.random_episodes(2, **sample_kwargs)
    rep = encode(agent, dummy, gpu_transform, args.device)
    in_dim = rep.shape[1]
    print(f"encoder rep dim (n_tokens * hidden_dim) = {in_dim}")
    print(f"  -> probe head: Linear({in_dim} -> {args.probe_hidden_dim}) "
          f"x3 hidden GELU -> Linear({args.probe_hidden_dim} -> {target_dim})")

    probe = ProbeMLP(in_dim, args.probe_hidden_dim, target_dim).to(args.device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    # ------------------------------------------------------------- train loop
    pbar = tqdm.trange(args.steps, desc="probe", dynamic_ncols=True)
    for step in pbar:
        batch = train_buf.random_episodes(args.batch_size, **sample_kwargs)
        rep = encode(agent, batch, gpu_transform, args.device)
        target = batch["target_now"].to(args.device)

        pred = probe(rep)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            with torch.no_grad():
                v_batch = val_buf.random_episodes(args.batch_size, **sample_kwargs)
                v_rep = encode(agent, v_batch, gpu_transform, args.device)
                v_target = v_batch["target_now"].to(args.device)
                v_pred = probe(v_rep)
                v_loss = F.mse_loss(v_pred, v_target).item()
                pos_err = (v_pred[:, :3] - v_target[:, :3]).norm(dim=1).mean().item()
                quat_err = (v_pred[:, 3:] - v_target[:, 3:]).norm(dim=1).mean().item()
            pbar.set_postfix(
                train=f"{loss.item():.4f}",
                val=f"{v_loss:.4f}",
                pos_l2=f"{pos_err:.4f}",
                quat_l2=f"{quat_err:.4f}",
            )

    if args.save_probe:
        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "in_dim": in_dim,
                "probe_hidden_dim": args.probe_hidden_dim,
                "target_dim": target_dim,
                "probe_keys": args.probe_keys,
                "ckpt": args.ckpt,
            },
            args.save_probe,
        )
        print(f"saved probe to {args.save_probe}")


if __name__ == "__main__":
    main()

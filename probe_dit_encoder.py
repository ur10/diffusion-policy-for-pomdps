"""Load a DiT BC checkpoint and report the shape of the output at the end of
the final transformer-encoder layer of the noise net.

Inside `_DiTNoiseNet` (policies/models/dit/models/diffusion_dit.py), the
"encoder" is the stack of `_SelfAttnEncoder` blocks that processes visual
(and optional proprio) tokens before they condition the action decoder.
`forward_enc` returns one tensor per block; `enc_cache[-1]` is the output
of the final block. This script runs a dummy forward pass and prints the
shape of that final-layer output.

Example:
    python probe_dit_encoder.py \\
        --ckpt logs/pusht/multi_pusht/2026-04-24-19:12:13+199282-199282/save/agent_04500_perf0.000.pt
"""

import argparse

import torch
from ml_collections import ConfigDict

from policies.models.bc_policy_dit import BCPolicy_DiT


def build_config(
    actor_lr=1e-4,
    encoder_name="resnet",
    encoder_size=18,
    norm_groups=16,
    avg_pool=True,
    n_cams=1,
    use_obs=False,
    obs_history=3,
    ac_chunk=16,
    train_diffusion_steps=100,
    eval_diffusion_steps=8,
    time_dim=256,
    hidden_dim=512,
    num_blocks=6,
    dropout=0.1,
    nhead=8,
    dim_feedforward=2048,
    activation="gelu",
    causal_encoder=False,
):
    cfg = ConfigDict()
    cfg.actor_lr = actor_lr

    cd = ConfigDict()
    cd.encoder = ConfigDict()
    cd.encoder.name = encoder_name
    cd.encoder.size = encoder_size
    cd.encoder.weights = None
    cd.encoder.restore_path = ""
    cd.encoder.norm_cfg = ConfigDict()
    cd.encoder.norm_cfg.name = "group_norm"
    cd.encoder.norm_cfg.num_groups = norm_groups
    cd.encoder.avg_pool = avg_pool
    cd.n_cams = n_cams
    cd.use_obs = use_obs
    cd.obs_history = obs_history
    cd.ac_chunk = ac_chunk
    cd.train_diffusion_steps = train_diffusion_steps
    cd.eval_diffusion_steps = eval_diffusion_steps

    nn_kw = ConfigDict()
    nn_kw.time_dim = time_dim
    nn_kw.hidden_dim = hidden_dim
    nn_kw.num_blocks = num_blocks
    nn_kw.dropout = dropout
    nn_kw.nhead = nhead
    nn_kw.dim_feedforward = dim_feedforward
    nn_kw.activation = activation
    nn_kw.causal_encoder = causal_encoder
    cd.noise_net_kwargs = nn_kw

    cfg.config_diffusion = cd
    return cfg


@torch.no_grad()
def probe(ckpt_path, obs_dim, ac_dim, img_size, batch, device, **cfg_overrides):
    config_rl = build_config(**cfg_overrides)
    cd = config_rl.config_diffusion

    policy = BCPolicy_DiT(
        obs_dim=obs_dim,
        action_dim=ac_dim,
        config_seq=None,
        config_rl=config_rl,
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"loaded: {ckpt_path}")
    print(f"  missing keys ({len(missing)}): "
          f"{missing[:3]}{' ...' if len(missing) > 3 else ''}")
    print(f"  unexpected keys ({len(unexpected)}): "
          f"{unexpected[:3]}{' ...' if len(unexpected) > 3 else ''}")

    policy.eval()
    agent = policy.agent  # underlying DiffusionTransformerAgent (EMA-swapped in eval)

    obs = torch.zeros(batch, cd.obs_history, obs_dim, device=device)
    imgs = {
        f"cam{c}": torch.zeros(batch, cd.obs_history, 3, img_size, img_size, device=device)
        for c in range(cd.n_cams)
    }

    s_t = agent.tokenize_obs(imgs, obs)            # (B, n_tokens, token_dim)
    enc_cache = agent.noise_net.forward_enc(s_t)   # list of (n_tokens, B, hidden_dim)
    final = enc_cache[-1]
    n_tokens, B, hidden_dim = final.shape

    print()
    print(f"tokenize_obs output:           shape={tuple(s_t.shape)}  "
          f"(B, n_tokens, token_dim)")
    print(f"# transformer-encoder blocks:  {len(enc_cache)}")
    print(f"final encoder-layer output:    shape={tuple(final.shape)}  "
          f"(n_tokens, B, hidden_dim)")
    print(f"  -> n_tokens   = {n_tokens}")
    print(f"  -> batch      = {B}")
    print(f"  -> hidden_dim = {hidden_dim}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to agent_*.pt checkpoint")
    p.add_argument("--obs_dim", type=int, default=5,
                   help="proprio obs dim (PushT=5, robomimic varies)")
    p.add_argument("--ac_dim", type=int, default=2,
                   help="action dim (PushT=2)")
    p.add_argument("--img_size", type=int, default=96,
                   help="square image edge length (PushT=96, robomimic=84/256)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # config overrides — must match the training config behind the checkpoint
    p.add_argument("--obs_history", type=int, default=3)
    p.add_argument("--ac_chunk", type=int, default=16)
    p.add_argument("--n_cams", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--encoder_size", type=int, default=18)
    p.add_argument("--avg_pool", action="store_true", default=True)
    p.add_argument("--no_avg_pool", dest="avg_pool", action="store_false")
    p.add_argument("--use_obs", default=False,
                   help='False, "add_token", or "pad_img_tokens"')

    args = p.parse_args()

    use_obs = args.use_obs
    if isinstance(use_obs, str) and use_obs.lower() in ("false", "0", "none", ""):
        use_obs = False

    probe(
        ckpt_path=args.ckpt,
        obs_dim=args.obs_dim,
        ac_dim=args.ac_dim,
        img_size=args.img_size,
        batch=args.batch,
        device=args.device,
        obs_history=args.obs_history,
        ac_chunk=args.ac_chunk,
        n_cams=args.n_cams,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        nhead=args.nhead,
        encoder_size=args.encoder_size,
        avg_pool=args.avg_pool,
        use_obs=use_obs,
    )


if __name__ == "__main__":
    main()

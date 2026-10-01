"""
Visualize encoder self-attention scores across observation chunks during rollout.

At each timestep t, an observation chunk of `obs_history` frames is fed to the
transformer encoder. This script captures the self-attention scores between
frames in each chunk and visualizes how attention patterns evolve over time.

Example with obs_history=3:
    t=0: chunk = [f0, f0, f0]  → attention matrix (3x3)
    t=1: chunk = [f0, f0, f1]  → attention matrix (3x3)
    t=2: chunk = [f0, f1, f2]  → attention matrix (3x3)
    t=3: chunk = [f1, f2, f3]  → attention matrix (3x3)
    ...

Usage:
    python visualize_attention.py \
        --config_env=configs/envs/robomimic_can.py \
        --config_rl=configs/rl/bc_diffusion_dit.py \
        --checkpoint=/path/to/checkpoint.pt \
        --n_rollouts=1 \
        --output_dir=./attention_vis

"""
import os

os.environ["MUJOCO_GL"] = "egl"

from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from absl import app, flags
from ml_collections import config_flags

from torchkit.pytorch_utils import set_gpu_mode
from torchkit import pytorch_utils as ptu
from policies.models import AGENT_CLASSES
from policies.models.dit.transforms import get_transform_by_name
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env", None, "Environment configuration.", lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_rl", None, "RL algorithm configuration.", lock_config=False,
)

flags.DEFINE_string("checkpoint", None, "Path to checkpoint.")
flags.DEFINE_integer("n_rollouts", 1, "Number of rollouts to visualize.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("output_dir", "./attention_vis", "Output directory.")
flags.DEFINE_integer("max_steps", 100, "Max steps per rollout.")
flags.DEFINE_integer("encoder_layer", -1, "Which encoder layer to visualize (-1 = last layer).")
flags.DEFINE_boolean("debug", False, "Print debug information.")

flags.mark_flags_as_required(["config_rl", "config_env", "checkpoint"])


# Global storage for attention weights captured via hooks
ATTENTION_WEIGHTS = []
_STEP_COUNTER = [0]  # Mutable counter for debugging


def attention_hook(module, input, output):
    """Hook to capture attention weights from MultiheadAttention."""
    # output = (attn_output, attn_weights) when need_weights=True
    if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
        ATTENTION_WEIGHTS.append(output[1].detach().cpu())


def patch_encoder_attention(model, debug=False):
    """
    Patch encoder self-attention layers to return attention weights.
    Returns the number of encoder layers patched.
    """
    # Check if encoder exists and is used
    if not hasattr(model, 'noise_net'):
        print("Warning: No noise_net found in model")
        return 0

    noise_net = model.noise_net
    if not hasattr(noise_net, '_use_encoder'):
        print("Warning: noise_net doesn't have _use_encoder attribute")
    else:
        print(f"noise_net._use_encoder = {noise_net._use_encoder}")
        if not noise_net._use_encoder:
            print("ERROR: use_encoder=False - encoder is BYPASSED, no attention to capture!")
            print("The model uses external features (e.g., V-JEPA) without transformer encoder.")
            return 0

    if not hasattr(noise_net, 'encoder'):
        print("Warning: No encoder found in noise_net")
        return 0

    encoder = noise_net.encoder
    n_layers = len(encoder.layers)

    for layer_idx, layer in enumerate(encoder.layers):
        original_forward = layer.forward

        def make_patched_forward(layer_ref, orig_fwd, lidx, dbg):
            def patched_forward(src, pos, key_padding_mask=None):
                from policies.models.dit.models.diffusion_dit import _with_pos_embed

                # Debug: confirm encoder is being called
                if dbg and lidx == 0:
                    _STEP_COUNTER[0] += 1
                    print(f"\n  [Encoder called, step #{_STEP_COUNTER[0]}]")
                    print(f"  Input src shape: {src.shape} (seq_len, batch, dim)")
                    print(f"  Input src hash: {hash(src.cpu().numpy().tobytes()) % 10000}")  # Track if input changes

                q = k = _with_pos_embed(src, pos)
                attn_mask = None
                if layer_ref._causal:
                    T = src.shape[0]
                    attn_mask = torch.triu(torch.ones(T, T, device=src.device), diagonal=1).bool()

                # Capture attention weights (don't average across heads to see full pattern)
                src2, attn_weights = layer_ref.self_attn(
                    q, k, value=src,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,  # Get per-head attention
                )

                # attn_weights shape: (batch, num_heads, tgt_len, src_len)
                # Average across heads manually
                attn_weights_avg = attn_weights.mean(dim=1)  # (batch, tgt_len, src_len)

                if dbg and lidx == 0:
                    print(f"  Attention shape: {attn_weights.shape} (batch, heads, tgt, src)")
                    print(f"  Attention[0,0,:3,:3]:\n{attn_weights[0, 0, :3, :3]}")
                    print(f"  Attention avg[0,:3,:3]:\n{attn_weights_avg[0, :3, :3]}")

                # Store in global
                ATTENTION_WEIGHTS.append(attn_weights_avg.detach().cpu())

                # Rest of forward
                src = src + layer_ref.dropout1(src2)
                src = layer_ref.norm1(src)
                src2 = layer_ref.linear2(layer_ref.dropout2(layer_ref.activation(layer_ref.linear1(src))))
                src = src + layer_ref.dropout3(src2)
                src = layer_ref.norm2(src)
                return src

            return patched_forward

        layer.forward = make_patched_forward(layer, original_forward, layer_idx, debug)

    print(f"Patched {n_layers} encoder layers for attention capture")
    return n_layers


def aggregate_attention_to_frames(
    attn_weights: torch.Tensor,
    obs_history: int,
    n_tokens_per_frame: int,
    debug: bool = False,
) -> np.ndarray:
    """
    Aggregate token-level attention to frame-level attention.

    Args:
        attn_weights: (B, n_tokens, n_tokens) or (n_tokens, n_tokens)
        obs_history: Number of frames in observation chunk
        n_tokens_per_frame: Tokens per frame (1 if avg_pool=True, else e.g. 49)
        debug: Print debug information

    Returns:
        frame_attn: (obs_history, obs_history) attention matrix between frames
    """
    if debug:
        print(f"  aggregate_attention: input shape={attn_weights.shape}, obs_history={obs_history}, tokens_per_frame={n_tokens_per_frame}")

    if attn_weights.dim() == 3:
        attn_weights = attn_weights[0]  # Take first batch

    attn = attn_weights.numpy()
    n_tokens = attn.shape[0]

    if debug:
        print(f"  aggregate_attention: n_tokens={n_tokens}, attn min={attn.min():.4f}, max={attn.max():.4f}")
        print(f"  aggregate_attention: row sums (should be ~1): {attn.sum(axis=1)[:5]}")

    # If only 1 token total, can't visualize meaningful attention
    if n_tokens == 1:
        print("  WARNING: Only 1 token - attention is trivially 1.0")
        return np.ones((obs_history, obs_history))

    # If 1 token per frame and n_tokens matches obs_history, attention is frame-level
    if n_tokens_per_frame == 1 and n_tokens >= obs_history:
        frame_attn = attn[:obs_history, :obs_history]
        if debug:
            print(f"  aggregate_attention: using direct frame attention, shape={frame_attn.shape}")
        return frame_attn

    # Otherwise, aggregate tokens to frames
    # Tokens are ordered: [frame0_tok0, ..., frame0_tokN, frame1_tok0, ..., frame1_tokN, ...]
    n_img_tokens = obs_history * n_tokens_per_frame

    if n_tokens >= n_img_tokens:
        img_attn = attn[:n_img_tokens, :n_img_tokens]
        # Reshape to (n_frames, toks_per_frame, n_frames, toks_per_frame)
        img_attn = img_attn.reshape(obs_history, n_tokens_per_frame, obs_history, n_tokens_per_frame)
        # Average across tokens within each frame
        frame_attn = img_attn.mean(axis=(1, 3))
        if debug:
            print(f"  aggregate_attention: aggregated from {n_img_tokens} tokens to {obs_history}x{obs_history} frames")
        return frame_attn

    # Fallback: return raw attention (may not be frame-aligned)
    if debug:
        print(f"  aggregate_attention: WARNING - token count mismatch, returning raw attention")
    return attn[:obs_history, :obs_history] if n_tokens >= obs_history else attn


def get_obs_and_action_dims(env):
    assert len(env.observation_space.shape) == 1
    obs_dim = env.observation_space.shape[0]
    if env.action_space.__class__.__name__ == "Box":
        act_dim = env.action_space.shape[0]
    else:
        act_dim = env.action_space.n
    return obs_dim, act_dim


def build_agent(config_rl, obs_dim, act_dim):
    if config_rl.algo == "bc_diffusion_dit":
        agent_class = AGENT_CLASSES["Policy_BC_DiT"]
    elif config_rl.algo == "bc_diffusion":
        agent_class = AGENT_CLASSES["Policy_BC_Diffusion"]
    elif config_rl.algo in ("bc_ptp", "bc_ldp"):
        agent_class = AGENT_CLASSES["Policy_BC_PTP"]
    else:
        raise ValueError(f"Unsupported algo: {config_rl.algo}")

    from ml_collections import ConfigDict
    agent = agent_class(
        obs_dim=obs_dim,
        action_dim=act_dim,
        config_seq=ConfigDict(),
        config_rl=config_rl,
        image_encoder_fn=lambda: None,
        freeze_critic=False,
    )
    return agent


def run_rollout_and_collect_attention(
    config_env,
    config_rl,
    agent,
    device,
    max_steps,
    encoder_layer,
):
    """
    Run rollout and collect attention weights at each timestep.

    Returns:
        frames_per_step: List of (obs_history,) lists of frames at each step
        attention_per_step: List of (obs_history, obs_history) attention matrices
        rewards: List of rewards
        success: bool
    """
    global ATTENTION_WEIGHTS

    from envs.robomimic.robosuite import (
        _render, _obs_dict_to_vec, _denormalize_action, _MAX_STEPS, EXP_WEIGHT,
        load_action_stats, make_env as make_robomimic_env
    )

    load_action_stats(config_env.env_name, config_env.datadir)

    cfg = config_rl.config_diffusion
    obs_history = cfg.get("obs_history", 1)

    inner_agent = agent.agent
    n_tokens_per_frame = inner_agent._n_img_toks_per_frame
    if n_tokens_per_frame == 0:
        n_tokens_per_frame = 1

    # Debug: print token info
    print(f"\n=== Token Configuration ===")
    print(f"  obs_history: {obs_history}")
    print(f"  n_cams: {inner_agent._n_cams}")
    print(f"  n_tokens_per_frame (image): {n_tokens_per_frame}")
    print(f"  total n_tokens (from agent): {inner_agent._n_tokens}")
    print(f"  obs_strategy: {inner_agent._obs_strat}")
    print(f"  token_dim: {inner_agent._token_dim}")

    # Calculate expected tokens
    n_img_tokens = obs_history * n_tokens_per_frame * inner_agent._n_cams
    n_obs_tokens = obs_history if inner_agent._obs_strat == "add_token" else 0
    expected_total = n_img_tokens + n_obs_tokens
    print(f"  Expected tokens: {n_img_tokens} img + {n_obs_tokens} obs = {expected_total}")
    print(f"===========================\n")

    n_encoder_layers = patch_encoder_attention(inner_agent, debug=True)
    if n_encoder_layers == 0:
        raise RuntimeError("No encoder layers found to patch")

    # Determine which layer to use
    layer_idx = encoder_layer if encoder_layer >= 0 else n_encoder_layers - 1
    layer_idx = min(layer_idx, n_encoder_layers - 1)
    print(f"Using encoder layer {layer_idx} (of {n_encoder_layers})")

    # Setup env
    env = make_robomimic_env(config_env.env_name, config_env.datadir)
    transform = get_transform_by_name("preproc")

    o = env.reset()
    o_vec = _obs_dict_to_vec(o)
    raw_img = _render(env)
    img_t = transform(torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255)

    obs_deque = deque([torch.from_numpy(o_vec)] * obs_history, maxlen=obs_history)
    img_deque = deque([img_t] * obs_history, maxlen=obs_history)
    raw_img_deque = deque([raw_img.copy()] * obs_history, maxlen=obs_history)

    done = False
    t = 0
    total_reward = 0
    act_history = None
    executed_action_history = deque(maxlen=max(obs_history - 1, 1))

    # Storage
    frames_per_step = []
    attention_per_step = []
    rewards = []

    effective_max_steps = min(max_steps, _MAX_STEPS)

    while not done and t < effective_max_steps and not env.is_success()["task"]:
        # Clear global attention storage
        ATTENTION_WEIGHTS.clear()

        # Prepare inputs
        obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(device)
        imgs_stacked = torch.stack(list(img_deque), dim=0)[None].to(device)
        imgs = {"cam0": imgs_stacked}

        # Debug: Check if image inputs are changing
        if t < 3 or t % 20 == 0:
            img_hash = hash(imgs_stacked.cpu().numpy().tobytes()) % 10000
            print(f"\n[Step {t}] Image tensor hash: {img_hash}")

        # Inference - this populates ATTENTION_WEIGHTS
        with torch.no_grad():
            acs = inner_agent.get_actions(imgs, obs_stacked, obs_valid_len=obs_history)[0]

        # Debug: Check how many attention matrices were captured
        if t < 3:
            print(f"[Step {t}] Captured {len(ATTENTION_WEIGHTS)} attention matrices")

        # Extract attention from selected layer
        if len(ATTENTION_WEIGHTS) > layer_idx:
            attn = ATTENTION_WEIGHTS[layer_idx]

            # Debug: Compare attention from different layers
            if t < 3:
                print(f"\n[Step {t}] Comparing attention across {len(ATTENTION_WEIGHTS)} layers:")
                for li in range(min(len(ATTENTION_WEIGHTS), 3)):  # First 3 layers
                    layer_attn = ATTENTION_WEIGHTS[li][0]  # First batch
                    print(f"  Layer {li}: [0,0]={layer_attn[0,0]:.4f}, [0,1]={layer_attn[0,1]:.4f}, [1,0]={layer_attn[1,0]:.4f}")
                last_li = len(ATTENTION_WEIGHTS) - 1
                last_attn = ATTENTION_WEIGHTS[last_li][0]
                print(f"  Layer {last_li} (LAST): [0,0]={last_attn[0,0]:.4f}, [0,1]={last_attn[0,1]:.4f}, [1,0]={last_attn[1,0]:.4f}")
                print(f"  Using layer_idx={layer_idx}")

            frame_attn = aggregate_attention_to_frames(attn, obs_history, n_tokens_per_frame, debug=(t == 0))
        else:
            print(f"  WARNING: No attention captured (got {len(ATTENTION_WEIGHTS)} weights, need layer {layer_idx})")
            frame_attn = np.zeros((obs_history, obs_history))

        # Store frames and attention
        frames_per_step.append([img.copy() for img in raw_img_deque])
        attention_per_step.append(frame_attn)

        # Debug: verify attention is changing
        if t < 5 or t % 20 == 0:
            print(f"[Step {t}] Stored frame_attn shape={frame_attn.shape}, diagonal={np.diag(frame_attn)}")

        # Execute action
        acs_np = acs.cpu().numpy()
        if act_history is None:
            act_history = deque(maxlen=len(acs_np))
        act_history.append(acs_np)

        num_actions = len(act_history)
        curr_act_preds = np.stack([
            pred_actions[idx]
            for idx, pred_actions in zip(range(num_actions - 1, -1, -1), act_history)
        ])
        weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))[::-1]
        weights = weights / weights.sum()
        ac = np.sum(weights[:, None] * curr_act_preds, axis=0)
        ac = _denormalize_action(ac)

        o, r, done, _ = env.step(ac)
        t += 1
        total_reward += r
        rewards.append(r)
        executed_action_history.append(torch.from_numpy(ac).float())

        # Update history
        o_vec = _obs_dict_to_vec(o)
        raw_img = _render(env)
        img_t = transform(torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255)
        obs_deque.append(torch.from_numpy(o_vec))
        img_deque.append(img_t)
        raw_img_deque.append(raw_img.copy())

        print(f"\rStep {t}/{effective_max_steps}", end="")

    print()
    success = float(env.is_success()["task"])
    print(f"Rollout complete: steps={t}, reward={total_reward:.2f}, success={bool(success)}")

    # Debug: Verify attention varies across steps
    if len(attention_per_step) > 1:
        all_attn = np.stack(attention_per_step, axis=0)  # (n_steps, obs_history, obs_history)
        attn_std = all_attn.std(axis=0)  # std across time for each frame pair
        attn_range = all_attn.max(axis=0) - all_attn.min(axis=0)
        print(f"\n=== Attention Variation Across Steps ===")
        print(f"  Attention std (per position):\n{attn_std}")
        print(f"  Attention range (max-min per position):\n{attn_range}")
        print(f"  Mean attention across all steps:\n{all_attn.mean(axis=0)}")
        print(f"=========================================\n")

    return frames_per_step, attention_per_step, rewards, success


def create_visualization(
    frames_per_step: List[List[np.ndarray]],
    attention_per_step: List[np.ndarray],
    obs_history: int,
    output_dir: str,
    rollout_idx: int,
):
    """
    Create visualization showing frames and attention across all timesteps.
    """
    n_steps = len(frames_per_step)
    if n_steps == 0:
        print("No steps to visualize")
        return

    rollout_dir = os.path.join(output_dir, f"rollout_{rollout_idx:02d}")
    os.makedirs(rollout_dir, exist_ok=True)

    # 1. Save individual step visualizations
    print("Saving per-step visualizations...")
    for t in range(n_steps):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: frames in observation chunk
        frames = frames_per_step[t]
        stacked = np.concatenate(frames, axis=1)
        axes[0].imshow(stacked)
        axes[0].set_title(f"Observation Chunk at t={t}\n(oldest → newest)", fontsize=10)

        # Add frame boundaries
        h, w = frames[0].shape[:2]
        for i in range(1, len(frames)):
            axes[0].axvline(x=i * w, color='white', linewidth=2)
        # Add labels
        for i in range(len(frames)):
            label = f"t-{obs_history - 1 - i}"
            axes[0].text(i * w + w // 2, -5, label, ha='center', fontsize=9, color='black')
        axes[0].axis('off')

        # Right: attention heatmap
        attn = attention_per_step[t]
        im = axes[1].imshow(attn, cmap='hot', vmin=0, aspect='equal')
        axes[1].set_title(f"Frame-to-Frame Attention (t={t})", fontsize=10)
        axes[1].set_xlabel("Key frame (attended to)", fontsize=9)
        axes[1].set_ylabel("Query frame (attending)", fontsize=9)

        tick_labels = [f"t-{obs_history - 1 - i}" for i in range(obs_history)]
        axes[1].set_xticks(range(obs_history))
        axes[1].set_yticks(range(obs_history))
        axes[1].set_xticklabels(tick_labels, fontsize=8)
        axes[1].set_yticklabels(tick_labels, fontsize=8)

        # Add attention values as text
        for i in range(obs_history):
            for j in range(obs_history):
                axes[1].text(j, i, f"{attn[i, j]:.2f}", ha='center', va='center',
                           fontsize=8, color='white' if attn[i, j] < 0.5 else 'black')

        plt.colorbar(im, ax=axes[1], fraction=0.046)
        plt.tight_layout()
        # Use fixed figure size (no bbox_inches='tight') for consistent video frames
        plt.savefig(os.path.join(rollout_dir, f"step_{t:04d}.png"), dpi=120, pad_inches=0.1)
        plt.close()

    # 2. Create summary plot: attention evolution over time
    print("Creating attention evolution summary...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Stack all attention matrices: (n_steps, obs_history, obs_history)
    all_attn = np.stack(attention_per_step, axis=0)

    # (a) Average attention matrix across all steps
    ax = axes[0, 0]
    mean_attn = all_attn.mean(axis=0)
    im = ax.imshow(mean_attn, cmap='hot', vmin=0)
    ax.set_title("Average Attention Across All Timesteps", fontsize=11)
    ax.set_xlabel("Key frame")
    ax.set_ylabel("Query frame")
    tick_labels = [f"t-{obs_history - 1 - i}" for i in range(obs_history)]
    ax.set_xticks(range(obs_history))
    ax.set_yticks(range(obs_history))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    for i in range(obs_history):
        for j in range(obs_history):
            ax.text(j, i, f"{mean_attn[i, j]:.2f}", ha='center', va='center',
                   fontsize=9, color='white' if mean_attn[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)

    # (b) Attention received by each frame position over time
    ax = axes[0, 1]
    # Sum columns to get "attention received" by each frame
    attn_received = all_attn.sum(axis=1)  # (n_steps, obs_history)
    # Normalize per step
    attn_received = attn_received / attn_received.sum(axis=1, keepdims=True)

    for i in range(obs_history):
        ax.plot(attn_received[:, i], label=f"t-{obs_history - 1 - i}", linewidth=1.5)
    ax.set_xlabel("Rollout Timestep")
    ax.set_ylabel("Normalized Attention Received")
    ax.set_title("Attention Received by Each Frame Position Over Time", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Heatmap: attention to newest frame (t-0) over time
    ax = axes[1, 0]
    # How much does each query frame attend to the newest frame (column -1)?
    attn_to_newest = all_attn[:, :, -1]  # (n_steps, obs_history)
    im = ax.imshow(attn_to_newest.T, cmap='viridis', aspect='auto')
    ax.set_xlabel("Rollout Timestep")
    ax.set_ylabel("Query Frame")
    ax.set_title("How Each Frame Attends to NEWEST Frame (t-0) Over Time", fontsize=11)
    ax.set_yticks(range(obs_history))
    ax.set_yticklabels([f"t-{obs_history - 1 - i}" for i in range(obs_history)])
    plt.colorbar(im, ax=ax)

    # (d) Heatmap: attention to oldest frame (t-(obs_history-1)) over time
    ax = axes[1, 1]
    attn_to_oldest = all_attn[:, :, 0]  # (n_steps, obs_history)
    im = ax.imshow(attn_to_oldest.T, cmap='viridis', aspect='auto')
    ax.set_xlabel("Rollout Timestep")
    ax.set_ylabel("Query Frame")
    ax.set_title(f"How Each Frame Attends to OLDEST Frame (t-{obs_history-1}) Over Time", fontsize=11)
    ax.set_yticks(range(obs_history))
    ax.set_yticklabels([f"t-{obs_history - 1 - i}" for i in range(obs_history)])
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    summary_path = os.path.join(rollout_dir, "attention_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_path}")

    # 3. Create video
    print("Creating video...")
    try:
        import imageio
        import glob

        files = sorted(glob.glob(os.path.join(rollout_dir, "step_*.png")))
        if files:
            # Load all frames and find the max dimensions
            raw_frames = [Image.open(f) for f in files]
            max_w = max(f.width for f in raw_frames)
            max_h = max(f.height for f in raw_frames)

            # Resize all frames to the same size (pad with white if needed)
            frames = []
            for img in raw_frames:
                if img.width != max_w or img.height != max_h:
                    # Create white background and paste image
                    new_img = Image.new('RGB', (max_w, max_h), (255, 255, 255))
                    new_img.paste(img, (0, 0))
                    frames.append(np.array(new_img))
                else:
                    frames.append(np.array(img.convert('RGB')))

            video_path = os.path.join(output_dir, f"rollout_{rollout_idx:02d}_attention.mp4")
            imageio.mimsave(video_path, frames, fps=10)
            print(f"Saved: {video_path}")
    except ImportError as e:
        print(f"Skipping video creation: {e}")


def main(argv):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    set_gpu_mode(torch.cuda.is_available())
    print(f"Device: {ptu.device}")

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_env, env_name = config_env.create_fn(config_env)

    obs_history = config_rl.config_diffusion.get("obs_history", 1)
    avg_pool = config_rl.config_diffusion.encoder.get("avg_pool", True)

    print(f"Environment: {env_name}")
    print(f"Algorithm: {config_rl.algo}")
    print(f"obs_history: {obs_history}")
    print(f"encoder.avg_pool: {avg_pool}")
    print(f"Checkpoint: {FLAGS.checkpoint}")

    # Check if attention visualization will be meaningful
    if obs_history == 1 and avg_pool:
        print("\n" + "="*60)
        print("WARNING: With obs_history=1 and avg_pool=True, there is only")
        print("1 token, so attention is trivially 1.0.")
        print("For meaningful attention visualization, use obs_history > 1")
        print("or avg_pool=False (49 tokens per frame).")
        print("="*60 + "\n")

    env = make_env(env_name, FLAGS.seed)
    obs_dim, act_dim = get_obs_and_action_dims(env)

    print("Building agent...")
    agent = build_agent(config_rl, obs_dim, act_dim)
    agent = agent.to(ptu.device)

    print("Loading checkpoint...")
    state_dict = torch.load(FLAGS.checkpoint, map_location=ptu.device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print("Checkpoint loaded.")

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    for i in range(FLAGS.n_rollouts):
        print(f"\n{'='*50}")
        print(f"Rollout {i + 1}/{FLAGS.n_rollouts}")
        print('='*50)

        frames, attention, rewards, success = run_rollout_and_collect_attention(
            config_env, config_rl, agent, ptu.device,
            FLAGS.max_steps, FLAGS.encoder_layer,
        )

        create_visualization(
            frames, attention, obs_history,
            FLAGS.output_dir, rollout_idx=i,
        )

    print(f"\nAll visualizations saved to: {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)

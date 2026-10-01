"""
Evaluation script for trained diffusion policies.

Usage:
    python eval.py \
        --config_env=configs/envs/robomimic_can.py \
        --config_rl=configs/rl/bc_diffusion_dit.py \
        --checkpoint=/path/to/agent_10000_perf0.950.pt \
        --n_rollouts=50

    # With video recording:
    python eval.py \
        --config_env=configs/envs/robomimic_can.py \
        --config_rl=configs/rl/bc_diffusion_dit.py \
        --checkpoint=/path/to/agent_10000_perf0.950.pt \
        --n_rollouts=10 \
        --save_video \
        --video_dir=./eval_videos

"""
import os

os.environ["MUJOCO_GL"] = "egl"

from collections import deque

import numpy as np
import torch
from absl import app, flags
from ml_collections import config_flags

from torchkit.pytorch_utils import set_gpu_mode
from torchkit import pytorch_utils as ptu
from policies.models import AGENT_CLASSES
from policies.models.dit.transforms import get_transform_by_name
from envs.make_env import make_env

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

flags.DEFINE_string("checkpoint", None, "Path to the trained checkpoint (.pt file).")
flags.DEFINE_integer("n_rollouts", 50, "Number of evaluation rollouts.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("save_video", False, "Save videos of rollouts.")
flags.DEFINE_string("video_dir", "./eval_videos", "Directory to save videos.")
flags.DEFINE_integer("video_fps", 20, "FPS for saved videos.")

flags.mark_flags_as_required(["config_rl", "config_env", "checkpoint"])


def get_obs_and_action_dims(env):
    """Extract observation and action dimensions from environment."""
    assert len(env.observation_space.shape) == 1
    obs_dim = env.observation_space.shape[0]

    if env.action_space.__class__.__name__ == "Box":
        act_dim = env.action_space.shape[0]
    else:
        act_dim = env.action_space.n

    return obs_dim, act_dim


def build_agent(config_rl, obs_dim, act_dim):
    """Instantiate the policy agent based on config."""
    if config_rl.algo == "bc_diffusion":
        agent_class = AGENT_CLASSES["Policy_BC_Diffusion"]
    elif config_rl.algo == "bc_diffusion_dit":
        agent_class = AGENT_CLASSES["Policy_BC_DiT"]
    elif config_rl.algo in ("bc_ptp", "bc_ldp"):
        agent_class = AGENT_CLASSES["Policy_BC_PTP"]
    elif config_rl.algo == "bc_diffusion_jepa":
        agent_class = AGENT_CLASSES["Policy_BC_JEPA"]
    else:
        raise ValueError(f"Unsupported algo for eval: {config_rl.algo}")

    # Dummy config_seq (not used by diffusion policies)
    from ml_collections import ConfigDict
    config_seq = ConfigDict()

    agent = agent_class(
        obs_dim=obs_dim,
        action_dim=act_dim,
        config_seq=config_seq,
        config_rl=config_rl,
        image_encoder_fn=lambda: None,
        freeze_critic=False,
    )
    return agent


def save_video(frames, path, fps=20):
    """Save frames as an MP4 video."""
    import imageio
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved video: {path}")


def rollout_robomimic_with_video(
    task, agent, device, n_rollouts, save_video_flag, video_dir, video_fps
):
    """Run robomimic rollouts with optional video recording."""
    from envs.robomimic.robosuite import (
        _render, _obs_dict_to_vec, _denormalize_action, _MAX_STEPS, EXP_WEIGHT
    )

    transform = get_transform_by_name("preproc")
    env = task.eval_env
    obs_history = task._obs_history
    consistency_samples = task._consistency_samples

    success_flags, total_rewards = [], []

    if save_video_flag:
        os.makedirs(video_dir, exist_ok=True)

    for i in range(n_rollouts):
        print(f"Rollout {i + 1}/{n_rollouts}", end="\r")
        o = env.reset()

        # Build initial obs/image
        o_vec = _obs_dict_to_vec(o)
        raw_img = _render(env)
        img_t = transform(
            torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255
        )

        # Initialize history
        obs_deque = deque([torch.from_numpy(o_vec)] * obs_history, maxlen=obs_history)
        img_deque = deque([img_t] * obs_history, maxlen=obs_history)

        done = False
        t = 0
        total_reward = 0
        act_history = None
        executed_action_history = deque(maxlen=max(obs_history - 1, 1))

        # Video frames
        frames = [raw_img] if save_video_flag else None

        while not done and t < _MAX_STEPS and not env.is_success()["task"]:
            obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(device)
            imgs_stacked = torch.stack(list(img_deque), dim=0)[None].to(device)
            imgs = {"cam0": imgs_stacked}

            past_actions = None
            if len(executed_action_history) > 0:
                past_actions = torch.stack(list(executed_action_history), dim=0)[None].to(device)

            with torch.no_grad():
                if getattr(agent, "supports_consistency_sampling", False):
                    acs = agent.get_actions(
                        imgs,
                        obs_stacked,
                        past_actions=past_actions,
                        n_samples=consistency_samples,
                    )[0]
                else:
                    acs = agent.get_actions(imgs, obs_stacked, obs_valid_len=obs_history)[0]

            acs = acs.cpu().numpy()
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

            # Update history
            o_vec = _obs_dict_to_vec(o)
            raw_img = _render(env)
            img_t = transform(
                torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255
            )
            obs_deque.append(torch.from_numpy(o_vec))
            img_deque.append(img_t)

            if save_video_flag:
                frames.append(raw_img)

            if done:
                break

        success = float(env.is_success()["task"])
        success_flags.append(success)
        total_rewards.append(float(total_reward))

        # Save video
        if save_video_flag and frames:
            status = "success" if success else "fail"
            video_path = os.path.join(video_dir, f"rollout_{i:03d}_{status}.mp4")
            save_video(frames, video_path, fps=video_fps)

        print(f"Rollout {i + 1}/{n_rollouts}: reward={total_reward:.2f}, success={bool(success)}")

    return np.mean(success_flags), np.mean(total_rewards)


def rollout_pusht_with_video(
    task, agent, device, n_rollouts, save_video_flag, video_dir, video_fps
):
    """Run PushT rollouts with optional video recording."""
    from envs.pusht.pusht_tasks import _denormalize_action, _obs_from_info, EXP_WEIGHT

    transform = get_transform_by_name("preproc")
    obs_history = task._obs_history
    consistency_samples = task._consistency_samples
    max_steps = task._max_steps
    rng = task._rng

    success_flags, total_rewards = [], []

    if save_video_flag:
        os.makedirs(video_dir, exist_ok=True)

    for i in range(n_rollouts):
        print(f"Rollout {i + 1}/{n_rollouts}", end="\r")

        # Create fresh env for each rollout (handles randomization)
        env = task._make_env(rng)
        o, info = env.reset()

        # Build initial obs/image
        o_vec = _obs_from_info(info)
        raw_img = env.render()
        img_t = transform(
            torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255
        )

        # Initialize history
        obs_deque = deque([torch.from_numpy(o_vec)] * obs_history, maxlen=obs_history)
        img_deque = deque([img_t] * obs_history, maxlen=obs_history)

        done = False
        t = 0
        total_reward = 0
        act_history = None
        executed_action_history = deque(maxlen=max(obs_history - 1, 1))

        # Video frames
        frames = [raw_img] if save_video_flag else None

        while not done and t < max_steps:
            obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(device)
            imgs_stacked = torch.stack(list(img_deque), dim=0)[None].to(device)
            imgs = {"cam0": imgs_stacked}

            past_actions = None
            if len(executed_action_history) > 0:
                past_actions = torch.stack(list(executed_action_history), dim=0)[None].to(device)

            with torch.no_grad():
                if getattr(agent, "supports_consistency_sampling", False):
                    acs = agent.get_actions(
                        imgs,
                        obs_stacked,
                        past_actions=past_actions,
                        n_samples=consistency_samples,
                    )[0]
                else:
                    acs = agent.get_actions(imgs, obs_stacked, obs_valid_len=obs_history)[0]

            acs = acs.cpu().numpy()
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

            o, r, done, truncated, info = env.step(ac)
            t += 1
            total_reward += r
            executed_action_history.append(torch.from_numpy(ac).float())

            # Update history
            o_vec = _obs_from_info(info)
            raw_img = env.render()
            img_t = transform(
                torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255
            )
            obs_deque.append(torch.from_numpy(o_vec))
            img_deque.append(img_t)

            if save_video_flag:
                frames.append(raw_img)

            if done or truncated:
                break

        # PushT uses coverage as success metric
        success = info.get("coverage", 0.0)
        success_flags.append(success)
        total_rewards.append(float(total_reward))

        # Save video
        if save_video_flag and frames:
            video_path = os.path.join(video_dir, f"rollout_{i:03d}_cov{success:.2f}.mp4")
            save_video(frames, video_path, fps=video_fps)

        print(f"Rollout {i + 1}/{n_rollouts}: reward={total_reward:.2f}, coverage={success:.2f}")

    return np.mean(success_flags), np.mean(total_rewards)


def build_diffusion_task(config_env, config_rl, obs_dim, act_dim):
    """Build the task object that handles rollouts."""
    cfg = config_rl.config_diffusion
    obs_history = cfg.get("obs_history", 1)
    consistency_samples = cfg.get("consistency_samples", 1)

    if config_env.env_type == "robomimic":
        from envs.robomimic.robosuite import RoboSuiteTask, load_action_stats

        # Load action normalization stats (required for denormalization during rollout)
        load_action_stats(config_env.env_name, config_env.datadir)

        cam_keys = list(cfg.get("cam_keys", ("agentview_image",)))
        assert cfg.n_cams == len(cam_keys), (
            f"config_diffusion.n_cams={cfg.n_cams} disagrees with "
            f"cam_keys={cam_keys}"
        )
        task = RoboSuiteTask(
            task=config_env.env_name,
            n_cams=cfg.n_cams,
            obs_dim=obs_dim,
            ac_dim=act_dim,
            obs_history=obs_history,
            consistency_samples=consistency_samples,
            dataset_path=config_env.datadir or None,
            cam_keys=cam_keys,
        )
    elif config_env.env_type == "pusht":
        from envs.pusht.pusht_tasks import PushTTask, load_action_stats

        # Load action normalization stats (required for denormalization during rollout)
        load_action_stats(config_env.env_name, config_env.datadir)

        task = PushTTask(
            task=config_env.env_name,
            n_cams=cfg.n_cams,
            obs_dim=obs_dim,
            ac_dim=act_dim,
            obs_history=obs_history,
            consistency_samples=consistency_samples,
            render_size=config_env.get("img_size", 96),
            dataset_path=config_env.datadir or None,
        )
    elif config_env.env_type == "aloha":
        from envs.aloha.aloha import AlohaTask, load_action_stats

        load_action_stats(config_env.env_name, config_env.datadir)

        cam_keys = list(cfg.get("cam_keys", ("top", "right_wrist")))
        assert cfg.n_cams == len(cam_keys), (
            f"config_diffusion.n_cams={cfg.n_cams} disagrees with cam_keys={cam_keys}"
        )
        task = AlohaTask(
            task=config_env.env_name,
            n_cams=cfg.n_cams,
            obs_dim=obs_dim,
            ac_dim=act_dim,
            obs_history=obs_history,
            consistency_samples=consistency_samples,
            dataset_path=config_env.datadir or None,
            cam_keys=cam_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported env_type: {config_env.env_type}")

    return task


def main(argv):
    # Set random seeds
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    # Setup device
    set_gpu_mode(torch.cuda.is_available())
    print(f"Using device: {ptu.device}")

    # Load configs
    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_env, env_name = config_env.create_fn(config_env)

    print(f"Environment: {env_name}")
    print(f"Algorithm: {config_rl.algo}")
    print(f"Checkpoint: {FLAGS.checkpoint}")
    print(f"Number of rollouts: {FLAGS.n_rollouts}")
    print(f"Save videos: {FLAGS.save_video}")
    if FLAGS.save_video:
        print(f"Video directory: {FLAGS.video_dir}")

    # Create a dummy env to get dimensions
    print("Creating environment...")
    env = make_env(env_name, FLAGS.seed)
    obs_dim, act_dim = get_obs_and_action_dims(env)
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

    # Build agent
    print("Building agent...")
    agent = build_agent(config_rl, obs_dim, act_dim)
    agent = agent.to(ptu.device)

    # Load checkpoint
    print(f"Loading checkpoint from {FLAGS.checkpoint}...")
    state_dict = torch.load(FLAGS.checkpoint, map_location=ptu.device)
    agent.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    # Build evaluation task
    print("Building evaluation task...")
    task = build_diffusion_task(config_env, config_rl, obs_dim, act_dim)

    # Run evaluation
    print(f"\nRunning {FLAGS.n_rollouts} evaluation rollouts...")
    agent.eval()

    if FLAGS.save_video:
        # Use custom rollout with video recording
        if config_env.env_type == "robomimic":
            mean_success, mean_reward = rollout_robomimic_with_video(
                task, agent.agent, ptu.device, FLAGS.n_rollouts,
                FLAGS.save_video, FLAGS.video_dir, FLAGS.video_fps
            )
        elif config_env.env_type == "pusht":
            mean_success, mean_reward = rollout_pusht_with_video(
                task, agent.agent, ptu.device, FLAGS.n_rollouts,
                FLAGS.save_video, FLAGS.video_dir, FLAGS.video_fps
            )
        else:
            raise NotImplementedError(f"Video not supported for {config_env.env_type}")
    else:
        # Use original rollout (faster, no video)
        mean_success, mean_reward = task.rollout(
            agent.agent,
            ptu.device,
            n_rollouts=FLAGS.n_rollouts,
        )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate: {mean_success:.4f} ({mean_success * 100:.2f}%)")
    print(f"Mean Reward:  {mean_reward:.4f}")
    print("=" * 50)

    if FLAGS.save_video:
        print(f"\nVideos saved to: {FLAGS.video_dir}")


if __name__ == "__main__":
    app.run(main)

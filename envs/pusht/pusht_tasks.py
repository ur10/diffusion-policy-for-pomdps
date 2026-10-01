import os
import random
from collections import deque

import numpy as np
import torch
import tqdm

from buffers.seq_robosuite_buffer import SeqRobosuiteBuffer
from envs.pusht.drawer_pusht import DrawerPusht
from envs.pusht.multi_push import MultiPushEnv
from envs.pusht.pusht_env import PushTEnv, PushTEnvMulti
from envs.pusht.replay_buffer import ReplayBuffer
from policies.models.dit.transforms import get_transform_by_name, get_gpu_transform_by_name

BUF_SHUFFLE_RNG = 3904767649
_N_ROLLOUTS = 50
EXP_WEIGHT = 0.0

TASK_CONFIGS = {
    "demo_pusht": {
        "env_cls": PushTEnv,
        "obs_dim": 5,
        "ac_dim": 2,
        "max_episode_steps": 300,
    },
    "demo_pusht_multi": {
        "env_cls": PushTEnvMulti,
        "obs_dim": 5,
        "ac_dim": 2,
        "max_episode_steps": 400,
    },
    "drawer_pusht": {
        "env_cls": DrawerPusht,
        "obs_dim": 5,
        "ac_dim": 2,
        "max_episode_steps": 300,
    },
    "multi_pusht": {
        "env_cls": MultiPushEnv,
        "obs_dim": 5,
        "ac_dim": 2,
        "max_episode_steps": 300,
    },
}

_AC_LOC = None
_AC_SCALE = None


def _normalize_actions(actions):
    loc = _AC_LOC[None].astype(actions.dtype)
    scale = _AC_SCALE[None].astype(actions.dtype)
    return (actions - loc) / scale


def _denormalize_action(actions):
    return actions * _AC_SCALE + _AC_LOC


def load_action_stats(task, dataset_path):
    """Load action normalization stats from dataset without loading all data."""
    global _AC_LOC, _AC_SCALE

    if _AC_LOC is not None and _AC_SCALE is not None:
        return  # Already loaded

    task_path = _get_task_path(dataset_path)
    replay_buffer = ReplayBuffer.create_from_path(task_path, mode="r")
    all_actions = []

    print(f"Loading action statistics from {task_path}...")
    for episode_idx in range(replay_buffer.n_episodes):
        episode = replay_buffer.get_episode(episode_idx, copy=True)
        actions = np.asarray(episode["action"]).astype(np.float32)
        all_actions.append(actions)

    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    _AC_LOC = (max_ac + min_ac) / 2
    _AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)
    print(f"Action stats loaded: loc={_AC_LOC}, scale={_AC_SCALE}")


def _get_task_path(dataset_path):
    assert dataset_path is not None, "Must provide dataset_path for PushT tasks."
    task_path = os.path.expanduser(dataset_path)
    assert os.path.exists(task_path), f"Missing task data at {task_path}"
    return task_path


def _obs_from_info(info):
    return np.concatenate([info["pos_agent"], info["block_pose"]]).astype(np.float32)


def _load_demos(task, dataset_path):
    global _AC_LOC, _AC_SCALE

    task_path = _get_task_path(dataset_path)
    replay_buffer = ReplayBuffer.create_from_path(task_path, mode="r")

    buffer = []
    all_actions = []

    print(f"Loading PushT demonstration data from {task_path}!")
    for episode_idx in tqdm.tqdm(range(replay_buffer.n_episodes)):
        episode = replay_buffer.get_episode(episode_idx, copy=True)
        images = np.asarray(episode["img"]).astype(np.uint8)
        observations = np.asarray(episode["state"]).astype(np.float32)
        actions = np.asarray(episode["action"]).astype(np.float32)

        assert len(images) == len(observations) == len(actions), (
            f"Episode {episode_idx} has mismatched lengths: "
            f"img={len(images)} state={len(observations)} action={len(actions)}"
        )

        buffer.append(
            {
                "images": images,
                "observations": observations,
                "actions": actions,
            }
        )
        all_actions.append(actions)

    assert len(buffer) > 0, f"No episodes found in {task_path}"

    all_actions = np.concatenate(all_actions, axis=0)
    max_ac = np.max(all_actions, axis=0)
    min_ac = np.min(all_actions, axis=0)
    _AC_LOC = (max_ac + min_ac) / 2
    _AC_SCALE = np.maximum((max_ac - min_ac) / 2, 1e-6)

    for episode in buffer:
        episode["actions"] = _normalize_actions(episode["actions"])

    print(f"built task {task} with {len(buffer)} episodes")
    return buffer


def _make_eval_env(task, render_size, rng):
    if task == "demo_pusht":
        return PushTEnv(render_size=render_size)
    if task == "drawer_pusht":
        return DrawerPusht(
            render_size=render_size,
            signal_pos_idx=int(rng.integers(0, 2)),
            signal_color_idx=int(rng.integers(0, 2)),
        )
    if task == "multi_pusht":
        return MultiPushEnv(render_size=render_size, signal_idx=int(rng.integers(0, 3)))
    if task == "demo_pusht_multi":
        return PushTEnvMulti(render_size=render_size)
    raise KeyError(f"Unknown PushT task {task}")


class PushTBuffer(SeqRobosuiteBuffer):
    def __init__(
        self,
        task,
        obs_history=1,
        ac_chunk=1,
        n_train_demos=200,
        transform=None,
        dataset_path=None,
    ):
        demos = _load_demos(task, dataset_path)
        assert len(demos) >= n_train_demos, "Not enough demos!"

        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(demos)
        demos = demos[:n_train_demos]

        first = demos[0]
        H, W, C = first["images"].shape[1:]
        obs_dim = first["observations"].shape[1]
        ac_dim = first["actions"].shape[1]
        max_size = sum(len(d["observations"]) for d in demos) + len(demos)

        super().__init__(
            max_replay_buffer_size=max_size,
            observation_dim=obs_dim,
            action_dim=ac_dim,
            img_height=H,
            img_width=W,
            img_channels=C,
            obs_history=obs_history,
            ac_chunk=ac_chunk,
        )

        self._transform = transform

        for demo in tqdm.tqdm(demos):
            self.add_episode(
                images=demo["images"],
                observations=demo["observations"],
                actions=demo["actions"],
            )

    def random_episodes(self, batch_size, skip_imgs=False, raw_imgs=False):
        return super().random_episodes(batch_size, transform=self._transform, skip_imgs=skip_imgs, raw_imgs=raw_imgs)


class PushTTask:
    def __init__(
        self,
        task,
        n_cams,
        obs_dim,
        ac_dim,
        obs_history=1,
        consistency_samples=1,
        render_size=96,
        dataset_path=None,
        transform_name="preproc",
    ):
        del dataset_path
        spec = TASK_CONFIGS[task]
        assert n_cams == 1, "Only support single camera PushT tasks."
        assert obs_dim == spec["obs_dim"]
        assert ac_dim == spec["ac_dim"]
        self.task = task
        self._obs_history = obs_history
        self._consistency_samples = consistency_samples
        self._render_size = render_size
        self._transform_name = transform_name
        self._max_episode_steps = spec["max_episode_steps"]

    def rollout(self, agent, device, n_rollouts=_N_ROLLOUTS, action_exec_horizon=1, n_video_rollouts=0):
        gpu_transform = get_gpu_transform_by_name(self._transform_name).to(device)
        obs_history = self._obs_history
        success_flags, total_rewards, videos = [], [], []
        rng = np.random.default_rng(0)

        def _query_agent(obs_deque, img_deque, executed_action_history):
            obs_stacked = torch.stack(list(obs_deque), dim=0)[None].to(device)
            imgs_stacked = gpu_transform(torch.stack(list(img_deque), dim=0)[None].to(device))
            imgs = {"cam0": imgs_stacked}
            past_actions = None
            if len(executed_action_history) > 0:
                past_actions = torch.stack(list(executed_action_history), dim=0)[None].to(device)
            with torch.no_grad():
                if getattr(agent, "supports_consistency_sampling", False):
                    return agent.get_actions(
                        imgs, obs_stacked,
                        past_actions=past_actions,
                        n_samples=self._consistency_samples,
                    )[0].cpu().numpy()
                else:
                    return agent.get_actions(imgs, obs_stacked)[0].cpu().numpy()

        for i in range(n_rollouts):
            print(f"Rollout {i}", end="\r")
            record = i < n_video_rollouts
            frames = []

            env = _make_eval_env(self.task, self._render_size, rng)
            env.seed(i)
            env.reset()

            raw_img = env.render(mode="rgb_array")
            if record:
                frames.append(raw_img)
            info = env._get_info()
            o_vec = _obs_from_info(info)
            raw_img_t = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255

            obs_deque = deque(
                [torch.from_numpy(o_vec)] * obs_history, maxlen=obs_history
            )
            img_deque = deque([raw_img_t] * obs_history, maxlen=obs_history)
            executed_action_history = deque(maxlen=max(obs_history - 1, 1))

            if action_exec_horizon > 1:
                chunk_step = action_exec_horizon  # trigger re-query on first step
                current_chunk = None
            else:
                act_history = None

            done = False
            t = 0
            total_reward = 0.0

            while not done and t < self._max_episode_steps:
                if action_exec_horizon > 1:
                    if chunk_step >= action_exec_horizon:
                        current_chunk = _query_agent(obs_deque, img_deque, executed_action_history)
                        chunk_step = 0
                    ac = _denormalize_action(current_chunk[chunk_step])
                    chunk_step += 1
                else:
                    acs = _query_agent(obs_deque, img_deque, executed_action_history)
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

                obs, reward, done, info = env.step(ac)
                total_reward += float(reward)
                t += 1

                executed_action_history.append(torch.from_numpy(ac).float())
                obs_deque.append(torch.from_numpy(_obs_from_info(info)))

                raw_img = env.render(mode="rgb_array")
                if record:
                    frames.append(raw_img)
                img_deque.append(torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255)

            success_flags.append(float(done))
            total_rewards.append(total_reward)
            if record:
                videos.append(np.stack(frames))  # (T, H, W, C)
            env.close()

        return np.mean(success_flags), np.mean(total_rewards), videos

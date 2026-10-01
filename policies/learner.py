import os
import time

import math
import numpy as np
import torch
from torch.nn import functional as F

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import logger


class Learner:
    def __init__(self, env, eval_env, FLAGS, config_rl, config_seq, config_env):
        self.train_env = env
        self.eval_env = eval_env
        self.FLAGS = FLAGS
        self.config_rl = config_rl
        self.config_seq = config_seq
        self.config_env = config_env

        self.init_env()

        self.init_agent()

        self.init_train()

    def init_env(
        self,
    ):
        # get action / observation dimensions
        assert len(self.train_env.observation_space.shape) == 1  # flatten
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]
        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
    ):
        # initialize agent
        if self.config_rl.algo == "bc_diffusion":
            agent_class = AGENT_CLASSES["Policy_BC_Diffusion"]
        elif self.config_rl.algo == "bc_diffusion_dit":
            agent_class = AGENT_CLASSES["Policy_BC_DiT"]
        elif self.config_rl.algo in ("bc_ptp", "bc_ldp"):
            agent_class = AGENT_CLASSES["Policy_BC_PTP"]
        elif self.config_rl.algo == "bc_diffusion_jepa":
            if "Policy_BC_JEPA" not in AGENT_CLASSES:
                raise ModuleNotFoundError(
                    "bc_diffusion_jepa requires the external JEPA codebase and "
                    "its Python package `src` to be available."
                )
            agent_class = AGENT_CLASSES["Policy_BC_JEPA"]
        elif self.config_rl.algo == "bc_diffusion_videomae":
            agent_class = AGENT_CLASSES["Policy_BC_VideoMAE"]
        else:  # bc
            assert self.config_rl.datadir, "BC algorithm requires datadir for loading expert data"
            agent_class = AGENT_CLASSES["Policy_BC_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if self.config_rl.algo == "bc" and self.config_seq.model.observ_embedder.name == "cnn":
            image_encoder_fn = lambda: ImageEncoder(
                image_shape=self.train_env.image_space.shape,
                normalize_pixel=(self.train_env.observation_space.dtype == np.uint8),
                **self.config_seq.model.observ_embedder.to_dict(),
            )
        else:
            image_encoder_fn = lambda: None

        print(f"  Instantiating {agent_class.__name__} ...")
        self.agent = agent_class(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            config_seq=self.config_seq,
            config_rl=self.config_rl,
            image_encoder_fn=image_encoder_fn,
            freeze_critic=self.FLAGS.freeze_critic,
        )
        print(f"  Moving agent to {ptu.device} ...")
        self.agent = self.agent.to(ptu.device)
        print(f"  Agent ready.")
        logger.log(self.agent)

    def _init_train_diffusion(self):
        from policies.models.dit.transforms import get_transform_by_name, get_gpu_transform_by_name

        cfg = self.config_rl.config_diffusion
        obs_history = cfg.get("obs_history", 1)
        transform_name = cfg.get("transform", "preproc")
        if transform_name == "ldp_preproc":
            train_transform = get_transform_by_name("ldp_preproc_train")
            eval_transform = get_transform_by_name("ldp_preproc_eval")
        else:
            train_transform = get_transform_by_name(transform_name)
            eval_transform = get_transform_by_name(transform_name)
        transform = train_transform

        try:
            self._gpu_transform = get_gpu_transform_by_name(transform_name).to(ptu.device)
        except NotImplementedError:
            self._gpu_transform = None

        if self.config_env.env_type == "robomimic":
            from envs.robomimic.robosuite import RoboSuiteBuffer, RoboSuiteTask

            cam_keys = list(cfg.get("cam_keys", ("agentview_image",)))
            assert cfg.n_cams == len(cam_keys), (
                f"config_diffusion.n_cams={cfg.n_cams} disagrees with "
                f"cam_keys={cam_keys}"
            )
            print(f"  Loading RoboSuiteBuffer (task={self.config_env.env_name}, obs_history={obs_history}) ...")
            self._diffusion_buffer = RoboSuiteBuffer(
                task=self.config_env.env_name,
                obs_history=obs_history,
                ac_chunk=cfg.ac_chunk,
                n_train_demos=self.config_env.get("n_train_demos", self.config_rl.get("n_train_demos", 200)),
                transform=train_transform,
                dataset_path=self.config_env.datadir or None,
                cam_keys=cam_keys,
            )
            self._diffusion_task = RoboSuiteTask(
                task=self.config_env.env_name,
                n_cams=cfg.n_cams,
                obs_dim=self.obs_dim,
                ac_dim=self.act_dim,
                obs_history=obs_history,
                consistency_samples=cfg.get("consistency_samples", 1),
                dataset_path=self.config_env.datadir or None,
                eval_transform=eval_transform,
                cam_keys=cam_keys,
                transform_name="ldp_preproc_eval" if transform_name == "ldp_preproc" else transform_name,
                render_size=self.config_env.get("img_size", 256),
            )
        elif self.config_env.env_type == "pusht":
            from envs.pusht.pusht_tasks import PushTBuffer, PushTTask

            self._diffusion_buffer = PushTBuffer(
                task=self.config_env.env_name,
                obs_history=obs_history,
                ac_chunk=cfg.ac_chunk,
                n_train_demos=self.config_rl.get("n_train_demos", 200),
                transform=transform,
                dataset_path=self.config_env.datadir or None,
            )
            self._diffusion_task = PushTTask(
                task=self.config_env.env_name,
                n_cams=cfg.n_cams,
                obs_dim=self.obs_dim,
                ac_dim=self.act_dim,
                obs_history=obs_history,
                consistency_samples=cfg.get("consistency_samples", 1),
                render_size=self.config_env.get("img_size", 96),
                dataset_path=self.config_env.datadir or None,
                transform_name=transform_name,
            )
        else:
            raise NotImplementedError(
                f"Unsupported diffusion env_type {self.config_env.env_type}"
            )
        print(
            f"  Loading diffusion buffer (task={self.config_env.env_name}, "
            f"obs_history={obs_history}) ..."
        )
        print(f"  Buffer ready: {len(self._diffusion_buffer)} samples.")
        self.n_env_steps_total = self.config_rl.get("train_steps", 200_000)
        logger.log(
            f"Diffusion buffer: {len(self._diffusion_buffer)} samples, "
            f"obs_history={obs_history}, training for {self.n_env_steps_total} steps"
        )

    def init_train(
        self,
    ):
        if self.config_rl.algo in ("bc_diffusion", "bc_diffusion_dit", "bc_ptp", "bc_ldp", "bc_diffusion_jepa", "bc_diffusion_videomae"):
            self._init_train_diffusion()
            return

        if self.config_rl.datadir:
            self.policy_storage = RAMEfficient_SeqReplayBuffer.load(self.config_rl.datadir)
        else:
            self.policy_storage = RAMEfficient_SeqReplayBuffer(
                max_replay_buffer_size=max(
                    int(self.config_rl.replay_buffer_size),
                    int(
                        self.config_rl.replay_buffer_num_episodes
                        * self.train_env.max_episode_steps
                    ),
                ),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=self.config_seq.sampled_seq_len,
                observation_type=self.train_env.observation_space.dtype,
            )

        total_rollouts = self.FLAGS.start_training + self.FLAGS.train_episodes
        self.n_env_steps_total = self.train_env.max_episode_steps * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._start_time = time.time()
        self._start_time_last = time.time()
        self._best_perf = -np.inf

    def _train_diffusion(self):
        """Training loop for bc_diffusion (no env rollouts, pure supervised)."""
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        self._start_training()

        if hasattr(self.agent, "precompute_frozen_cache"):
            self.agent.precompute_frozen_cache(self._diffusion_buffer, ptu.device)

        step = 0
        last_eval_step = 0
        log_interval = 500

        profile_enabled = getattr(self.config_rl, "profile_training", False)
        _profiler = None
        if profile_enabled:
            import torch.profiler
            import wandb

            def _log_profile_to_wandb(prof):
                avgs = prof.key_averages()
                rows = []
                for evt in sorted(avgs, key=lambda e: e.self_device_time_total, reverse=True)[:30]:
                    rows.append([
                        evt.key,
                        round(evt.self_device_time_total / 1e3, 3),  # ms
                        round(evt.self_cpu_time_total / 1e3, 3),
                        evt.count,
                    ])
                wandb.log({"profiler/top_ops": wandb.Table(
                    columns=["op", "cuda_ms", "cpu_ms", "count"],
                    data=rows,
                )})
                print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))

            _profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
                on_trace_ready=_log_profile_to_wandb,
                record_shapes=True,
                with_stack=True,
            )
            _profiler.start()
            print("Profiler started — capturing steps 5-9. Results will be logged to wandb.")

        print("Running initial eval rollout before training ...")
        self._log_diffusion(0)

        from concurrent.futures import ThreadPoolExecutor
        pbar = tqdm(total=self.n_env_steps_total, desc="training", dynamic_ncols=True)
        with ThreadPoolExecutor(max_workers=1) as executor:
            prefetch_future = executor.submit(self._sample_diffusion_batch)
            while step < self.n_env_steps_total:
                batch = prefetch_future.result()
                prefetch_future = executor.submit(self._sample_diffusion_batch)
                train_stats = self.agent.update(batch)
                step += 1
                if _profiler is not None:
                    _profiler.step()
                    if step >= 10:
                        _profiler.stop()
                        _profiler = None
                        print("Profiling done — results logged to wandb.")
                self._n_env_steps_total = step
                loss = train_stats.get("diffusion_loss", float("nan"))
                pbar.set_postfix(loss=f"{loss:.4f}")
                pbar.update(1)

                if step % log_interval == 0:
                    self.log_train_stats(train_stats)

                if step - last_eval_step >= self.config_rl.eval_steps:
                    last_eval_step = step
                    self._log_diffusion(step)
        pbar.close()

    def train(self):
        """
        training loop
        """
        if self.config_rl.algo in ("bc_diffusion", "bc_diffusion_dit", "bc_ptp", "bc_ldp","bc_diffusion_jepa", "bc_diffusion_videomae"):
            return self._train_diffusion()

        self._start_training()

        if self.FLAGS.start_training > 0:
            logger.log("Collecting initial pool of data..")
            if not self.config_rl.datadir:
                while (
                    self._n_env_steps_total
                    < self.FLAGS.start_training * self.train_env.max_episode_steps
                ):
                    self.collect_rollouts(
                        num_rollouts=1,
                        random_actions=True,
                    )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            train_stats = self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )
            self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            if self.config_rl.datadir:
                env_steps = self.train_env.max_episode_steps
                self._n_env_steps_total += env_steps
                self._n_rollouts_total += 1
            else:
                env_steps = self.collect_rollouts(num_rollouts=1)

            train_stats = self.update(
                int(math.ceil(self.FLAGS.updates_per_step * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.train_env.max_episode_steps
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.config_env.eval_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                self.log()

    @torch.no_grad()
    def collect_eval_rollouts(self, num_rollouts, random_actions=False, only_successful=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []

            if not random_actions:
                # get hidden state at timestep=0
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    action, internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=False,
                    )

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                steps += 1

                # NOTE: designed by env
                term = self.config_env.terminal_fn(self.train_env, done_rollout, info)

                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            # add collected sequence to buffer
            act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
            if not self.act_continuous:
                act_buffer = torch.argmax(act_buffer, dim=-1, keepdims=True)  # (L, 1)
            if only_successful:
                if ptu.get_numpy(torch.cat(rew_list, dim=0)).sum() < 1.0:
                    continue
            self.policy_storage.add_episode(
                observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                actions=ptu.get_numpy(act_buffer),  # (L, dim)
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                next_observations=ptu.get_numpy(
                    torch.cat(next_obs_list, dim=0)
                ),  # (L, dim)
            )
            print(
                f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
            )

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False, only_successful=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []

            if not random_actions:
                # get hidden state at timestep=0
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info(
                    self.config_seq.sampled_seq_len
                )

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    action, internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=False,
                    )

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                steps += 1

                # NOTE: designed by env
                term = self.config_env.terminal_fn(self.train_env, done_rollout, info)

                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            # add collected sequence to buffer
            act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
            if not self.act_continuous:
                act_buffer = torch.argmax(act_buffer, dim=-1, keepdims=True)  # (L, 1)

            self.policy_storage.add_episode(
                observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                actions=ptu.get_numpy(act_buffer),  # (L, dim)
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                next_observations=ptu.get_numpy(
                    torch.cat(next_obs_list, dim=0)
                ),  # (L, dim)
            )
            print(
                f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
            )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def _sample_diffusion_batch(self):
        cache_active = getattr(self.agent, "_frozen_cache", None) is not None
        use_gpu_transform = self._gpu_transform is not None and not cache_active
        batch = self._diffusion_buffer.random_episodes(
            self.FLAGS.batch_size,
            skip_imgs=cache_active,
            raw_imgs=use_gpu_transform,
        )

        imgs = {}
        for cam, v in batch["imgs"].items():
            v = v.to(ptu.device, non_blocking=True)  # uint8 if raw_imgs, else float32
            if use_gpu_transform:
                v = v.float().div_(255.0)
                v = self._gpu_transform(v)
            imgs[cam] = v

        out = {
            "imgs": imgs,
            "obs":  batch["obs"].to(ptu.device, non_blocking=True),
            "act":  batch["act"].to(ptu.device, non_blocking=True),
            "mask": batch["mask"].to(ptu.device, non_blocking=True),
        }
        if "buf_starts" in batch:
            out["buf_starts"] = batch["buf_starts"]  # keep on CPU for cache indexing
        if "act_seq" in batch:
            out["act_seq"] = batch["act_seq"].to(ptu.device)
        if "mask_seq" in batch:
            out["mask_seq"] = batch["mask_seq"].to(ptu.device)
        return out

    def sample_rl_batch(self, batch_size):
        # rnn: all items are (sampled_seq_len, B, dim)
        batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            if self.config_rl.algo in ("bc_diffusion", "bc_diffusion_dit", "bc_ptp", "bc_ldp", "bc_diffusion_jepa", "bc_diffusion_videomae"):
                batch = self._sample_diffusion_batch()
            else:
                batch = self.sample_rl_batch(self.FLAGS.batch_size)
            rl_losses = self.agent.update(batch)
            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, deterministic=True):
        if self.config_rl.algo in ("bc_diffusion", "bc_diffusion_dit", "bc_ptp", "bc_ldp","bc_diffusion_jepa", "bc_diffusion_videomae"):
            self.agent.eval()
            mean_success, mean_reward, _ = self._diffusion_task.rollout(
                self.agent.agent, ptu.device,
                n_rollouts=self.config_env.eval_episodes,
                action_exec_horizon=self.config_rl.get("action_exec_horizon", 1),
            )
            self.agent.train()

            n = self.config_env.eval_episodes
            returns_per_episode = np.full(n, mean_reward, dtype=float)
            success_rate = np.full(n, mean_success, dtype=float)
            total_steps = np.zeros(n, dtype=float)
            return returns_per_episode, success_rate, total_steps

        self.agent.eval()  # set to eval mode for deterministic dropout

        returns_per_episode = np.zeros(self.config_env.eval_episodes)
        success_rate = np.zeros(self.config_env.eval_episodes)
        total_steps = np.zeros(self.config_env.eval_episodes)

        for task_idx in range(self.config_env.eval_episodes):
            step = 0
            running_reward = 0.0
            done_rollout = False

            obs = ptu.from_numpy(self.eval_env.reset())  # reset
            obs = obs.reshape(1, obs.shape[-1])

            # assume initial reward = 0.0
            action, reward, internal_state = self.agent.get_initial_info(
                self.config_seq.sampled_seq_len
            )

            while not done_rollout:
                action, internal_state = self.agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                )

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )

                # add raw reward
                running_reward += reward.item()
                step += 1
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: obs <- next_obs
                obs = next_obs.clone()

            returns_per_episode[task_idx] = running_reward
            total_steps[task_idx] = step
            if "success" in info and info["success"] == True:  # keytodoor
                success_rate[task_idx] = 1.0

        self.agent.train()  # set it back to train
        print(f"steps: {self._n_env_steps_total} returns: {returns_per_episode.mean():.2f}")
        return returns_per_episode, success_rate, total_steps

    def _log_diffusion(self, step):
        import wandb
        n_video = self.config_rl.get("eval_video_rollouts", 2)
        self.agent.eval()
        mean_success, mean_reward, videos = self._diffusion_task.rollout(
            self.agent.agent, ptu.device,
            n_rollouts=self.config_env.eval_episodes,
            action_exec_horizon=self.config_rl.get("action_exec_horizon", 1),
            n_video_rollouts=n_video,
        )
        self.agent.train()
        if videos and wandb.run is not None:
            wandb.log(
                {f"eval/rollout_{i}": wandb.Video(v.transpose(0, 3, 1, 2), fps=20, format="mp4")
                 for i, v in enumerate(videos)},
                step=step, commit=False,
            )
        logger.record_step("env_steps", step)
        logger.record_tabular("success", mean_success)
        logger.record_tabular("return", mean_reward)
        logger.dump_tabular()
        self.save_model(step, mean_success)

    def log_train_stats(self, train_stats):
        logger.record_step("env_steps", self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular(k, v)
        ## gradient norms
        results = self.agent.report_grad_norm()
        for k, v in results.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

    def log(self):
        logger.record_step("env_steps", self._n_env_steps_total)
        returns_eval, success_rate_eval, total_steps_eval = self.evaluate()
        logger.record_tabular("return", np.mean(returns_eval))
        logger.record_tabular("success", np.mean(success_rate_eval))
        logger.record_tabular("length", np.mean(total_steps_eval))
        logger.record_tabular(
            "FPS",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        perf = np.mean(returns_eval)
        self.save_model(self._n_env_steps_total, perf)
        self._maybe_plot_attention_maps()
        return perf

    @torch.no_grad()
    def _maybe_plot_attention_maps(self):
        from policies.seq_models.gpt2_vanilla import GPT2

        if not (
            hasattr(self.agent, "actor")
            and isinstance(self.agent.actor.seq_model, GPT2)
        ):
            return

        actor = self.agent.actor
        actor.eval()

        # Sample one sequence from the replay buffer: (T, 1, dim)
        batch = ptu.np_to_pytorch_batch(self.policy_storage.random_episodes(1))
        observs = batch["obs"]  # (T, 1, obs_dim)
        actions = batch["act"]  # (T, 1, 1) discrete or (T, 1, act_dim)
        rewards = batch["rew"]  # (T, 1, 1)

        if not self.act_continuous:
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.act_dim
            ).float()  # (T, 1, act_dim)

        input_embeds = torch.cat(
            [
                actor.action_embedder(actions),
                actor.reward_embedder(rewards),
                actor.observ_embedder(observs),
            ],
            dim=-1,
        )  # (T, 1, hidden_size)

        save_dir = os.path.join(logger.get_dir(), "attention_maps")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"attention_{self._n_env_steps_total:0{self._digit()}d}.png"
        )
        actor.seq_model.plot_attention_maps(input_embeds, save_path=save_path)
        actor.train()

    def save_model(self, total_steps, perf):
        if perf < self._best_perf:
            return
        self._best_perf = perf
        save_dir = os.path.join(logger.get_dir(), "save")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"agent_{total_steps:0{self._digit()}d}_perf{perf:.3f}.pt",
        )
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)

    def _digit(self):
        # zero pad with total env steps
        return int(math.log10(self.n_env_steps_total) + 1)

"""Replay buffer for probe training. Subclass of `SeqRobosuiteBuffer` that
additionally stores a per-step probing target (e.g. ground-truth nut pose)
and returns the target at the LAST frame of each sampled obs_history window.

Designed to live alongside (not replace) the policy-training buffer — none
of the main training code imports this.
"""

import numpy as np
import torch

from buffers.seq_robosuite_buffer import SeqRobosuiteBuffer


class ProbeBuffer(SeqRobosuiteBuffer):
    """Same windowing/sampling as `SeqRobosuiteBuffer`, plus a parallel
    `(_max_size, target_dim)` array of per-step probe targets. After sampling
    a batch, returns `target_now[b] = targets[start_b + obs_history - 1]`,
    which is the ground-truth label at the "current frame" of each window.
    """

    def __init__(self, *args, target_dim, **kwargs):
        super().__init__(*args, **kwargs)
        assert target_dim > 0, "ProbeBuffer requires target_dim > 0"
        self._target_dim = target_dim
        self._targets = np.zeros((self._max_size, target_dim), dtype=np.float32)

    def add_episode(self, images, observations, actions, targets):
        assert targets is not None, "ProbeBuffer requires targets per episode"
        seq_len_raw = observations.shape[0]
        assert targets.shape == (seq_len_raw, self._target_dim), (
            f"targets {targets.shape} != ({seq_len_raw}, {self._target_dim})"
        )

        # mirror the parent's first-frame padding so the padded target array
        # has the same length as the padded obs/imgs/actions
        if self._obs_history > 1:
            n_pad = self._obs_history - 1
            targets = np.concatenate(
                [np.tile(targets[:1], (n_pad, 1)), targets], axis=0
            )

        # capture write position BEFORE parent advances self._top
        write_start = self._top
        super().add_episode(images, observations, actions)

        seq_len = targets.shape[0]   # padded length (matches what parent wrote)
        indices = np.arange(write_start, write_start + seq_len) % self._max_size
        self._targets[indices] = targets

    def random_episodes(self, batch_size, transform=None, skip_imgs=False, raw_imgs=False):
        starts = self._sample_indices(batch_size)
        batch = self._make_batch(starts, transform, skip_imgs, raw_imgs)
        last_idx = (np.asarray(starts) + self._obs_history - 1) % self._max_size
        batch["target_now"] = torch.from_numpy(
            self._targets[last_idx].astype(np.float32).copy()
        )
        return batch

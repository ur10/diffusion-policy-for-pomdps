# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset


# helper functions
_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((2, 0, 1)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()
BUF_SHUFFLE_RNG = 3904767649


class SimTaskReplayBuffer(Dataset):
    def __init__(
        self,
        task,
        transform=None,
        n_train_demos=200,
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        past_frames=0,
        dataset_path=None,
    ):

        # these tasks don't require conditioning so skip
        assert cam_indexes == [0], "only need 0th cam"
        assert goal_indexes == [], "no need for goal indexes"
        assert past_frames == 0, "past frames should be 0"

        buffer_data = self.load_buffer(task, dataset_path)
        assert len(buffer_data) >= n_train_demos, "Not enough demos!"

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(buffer_data)

        # take n_train_demos demos for training
        buffer_data = buffer_data[:n_train_demos]

        self.transform = transform
        self.s_a_mask = []
        for traj in tqdm.tqdm(buffer_data):
            imgs, obs, acs = traj["images"], traj["observations"], traj["actions"]
            assert len(obs) == len(acs) and len(acs) == len(
                imgs
            ), "All time dimensions must match!"

            # pad camera dimension if needed
            if len(imgs.shape) == 4:
                imgs = imgs[:, None]

            for t in range(len(imgs) - 1):
                i_t = {f"cam{c}": imgs[t, c] for c in range(imgs.shape[1])}
                o_t = obs[t]

                loss_mask = np.ones((ac_chunk,), dtype=np.float32)
                a_t = acs[t : t + ac_chunk]
                assert len(a_t) > 0
                if len(a_t) < ac_chunk:
                    missing = ac_chunk - len(a_t)

                    action_pad = np.zeros((missing, a_t.shape[-1])).astype(np.float32)
                    a_t = np.concatenate((a_t, action_pad), axis=0)
                    loss_mask[-missing:] = 0

                self.s_a_mask.append(((i_t, o_t), a_t, loss_mask))

    def load_buffer(self, task, dataset_path=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        (i_t, o_t), a_t, loss_mask = self.s_a_mask[idx]

        i_t = {k: _img_to_tensor(v) for k, v in i_t.items()}
        if self.transform is not None:
            i_t = {k: self.transform(v) for k, v in i_t.items()}

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
        assert (
            loss_mask.shape[0] == a_t.shape[0]
        ), "a_t and mask shape must be ac_chunk!"
        return (i_t, o_t), a_t, loss_mask

"""
Dataloader for probing experiments on a DiT-Block-style policy.

Returns per-timestep samples containing:
  - multi-camera images (global, left_wrist, right_wrist)
  - proprioception
  - language goal (raw string; tokenize inside the model)
  - probe target: object positions (regression)

Assumed HDF5 layout (one file, many trajectories):

  root/
    traj_0000/
      images/
        global       (T, H, W, 3) uint8
        left_wrist   (T, H, W, 3) uint8
        right_wrist  (T, H, W, 3) uint8
      proprio          (T, D_proprio) float32
      actions          (T, D_action)  float32       # not strictly needed for probing
      object_positions (T, N_obj, 3)  float32       # probe target
      goal             scalar string  (attr or dataset)
    traj_0001/
      ...

If your layout differs, the only thing to change is `_index_trajectories`
and the dataset key strings inside `__getitem__`.
"""

from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CAMERA_KEYS = ("global", "left_wrist", "right_wrist")


class ProbeDataset(Dataset):
    """Per-timestep dataset for probing.

    Each item is one timestep from one trajectory. A flat (traj_idx, t) index
    is built once at construction time, so __len__ is total timesteps.
    """

    def __init__(
        self,
        h5_path: str,
        image_size: int = 224,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
        camera_keys: tuple[str, ...] = CAMERA_KEYS,
    ):
        """
        Args:
            h5_path: path to the HDF5 file.
            image_size: resize images to (image_size, image_size). Set to None to skip.
            skip_first_n: drop the first N steps of each trajectory (e.g. memory burn-in).
            skip_last_n: drop the last N steps of each trajectory.
            camera_keys: which cameras to load.
        """
        self.h5_path = h5_path
        self.image_size = image_size
        self.camera_keys = camera_keys
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n

        # File handle is opened lazily per worker (HDF5 + fork safety).
        self._h5: Optional[h5py.File] = None

        # Build a flat (traj_name, t) index without keeping the file open.
        self._index: list[tuple[str, int]] = self._index_trajectories()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _index_trajectories(self) -> list[tuple[str, int]]:
        index: list[tuple[str, int]] = []
        with h5py.File(self.h5_path, "r") as f:
            traj_names = sorted(f.keys())
            for name in traj_names:
                grp = f[name]
                T = grp["proprio"].shape[0]
                start = self.skip_first_n
                end = T - self.skip_last_n
                if end <= start:
                    continue
                for t in range(start, end):
                    index.append((name, t))
        return index

    # ------------------------------------------------------------------
    # File handle (lazy, per-worker)
    # ------------------------------------------------------------------
    def _file(self) -> h5py.File:
        if self._h5 is None:
            # 'r' + libver='latest' is safe for read-only multi-worker access.
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        """uint8 (H, W, 3) -> float32 (3, H', W') in [0, 1]."""
        t = torch.from_numpy(img).float() / 255.0      # (H, W, 3)
        t = t.permute(2, 0, 1).contiguous()            # (3, H, W)
        if self.image_size is not None and (
            t.shape[-1] != self.image_size or t.shape[-2] != self.image_size
        ):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        traj_name, t = self._index[idx]
        f = self._file()
        grp = f[traj_name]

        # Images: one per camera, stacked into a dict to match policy input.
        images = {
            cam: self._process_image(grp[f"images/{cam}"][t])
            for cam in self.camera_keys
        }

        proprio = torch.from_numpy(grp["proprio"][t]).float()
        target = torch.from_numpy(grp["object_positions"][t]).float()  # (N_obj, 3)

        # Goal: stored either as a per-trajectory attribute or a dataset.
        if "goal" in grp.attrs:
            goal = str(grp.attrs["goal"])
        elif "goal" in grp:
            raw = grp["goal"][()]
            goal = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        else:
            goal = ""

        return {
            "images": images,                # dict[str, Tensor(3, H, W)]
            "proprio": proprio,              # Tensor(D_proprio,)
            "goal": goal,                    # str
            "target": target,                # Tensor(N_obj, 3)  -- probe target
            "traj": traj_name,               # for debugging / per-traj eval
            "t": t,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Collation: goals are strings, so we can't use the default collate.
# ----------------------------------------------------------------------
def probe_collate(batch: list[dict]) -> dict:
    images = {
        cam: torch.stack([b["images"][cam] for b in batch], dim=0)
        for cam in batch[0]["images"]
    }
    proprio = torch.stack([b["proprio"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    goals = [b["goal"] for b in batch]
    trajs = [b["traj"] for b in batch]
    ts = torch.tensor([b["t"] for b in batch], dtype=torch.long)
    return {
        "images": images,        # dict[str, (B, 3, H, W)]
        "proprio": proprio,      # (B, D_proprio)
        "goal": goals,           # list[str] of length B
        "target": target,        # (B, N_obj, 3)
        "traj": trajs,
        "t": ts,
    }


# ----------------------------------------------------------------------
# Convenience builder
# ----------------------------------------------------------------------
def build_probe_loader(
    h5_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: int = 224,
    skip_first_n: int = 0,
    skip_last_n: int = 0,
) -> DataLoader:
    ds = ProbeDataset(
        h5_path=h5_path,
        image_size=image_size,
        skip_first_n=skip_first_n,
        skip_last_n=skip_last_n,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=probe_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/probe_data.h5"
    if not os.path.exists(path):
        print(f"No file at {path}; pass a path as the first arg.")
        sys.exit(0)

    loader = build_probe_loader(path, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    for cam, t in batch["images"].items():
        print(f"images[{cam}]: {tuple(t.shape)} dtype={t.dtype}")
    print(f"proprio: {tuple(batch['proprio'].shape)}")
    print(f"target:  {tuple(batch['target'].shape)}")
    print(f"goals:   {batch['goal']}")
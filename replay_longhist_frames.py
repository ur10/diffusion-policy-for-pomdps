"""Decode stored agentview_image frames from the longhist hdf5 to MP4s.

Usage:
    python replay_longhist_frames.py --dataset /home/ur10/Downloads/longhistsquare100/demos.hdf5 \
        --output_dir replays --demos 0 1 2 --fps 20 --upsample 4

`--upsample` enlarges each 84x84 frame for easier viewing; 4 → 336x336.
"""
import argparse
import os

import h5py
import imageio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", default="replays")
    ap.add_argument("--demos", type=int, nargs="*", default=None,
                    help="Specific demo indices to replay. Default: all.")
    ap.add_argument("--cam_key", default="agentview_image",
                    help="Which obs/<key> to extract. Try robot0_eye_in_hand_image too.")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--upsample", type=int, default=4,
                    help="Integer upscale via nearest-neighbor (84*4=336).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(f["data"].keys(),
                           key=lambda k: int(k.split("_")[-1]))
        if args.demos is not None:
            demo_keys = [f"demo_{i}" for i in args.demos]

        for dk in demo_keys:
            imgs = f[f"data/{dk}/obs/{args.cam_key}"][()]   # (L, H, W, C)
            if args.upsample > 1:
                imgs = imgs.repeat(args.upsample, axis=1).repeat(args.upsample, axis=2)
            out = os.path.join(args.output_dir, f"{dk}_{args.cam_key}.mp4")
            imageio.mimsave(out, imgs.astype(np.uint8), fps=args.fps)
            print(f"wrote {out}  ({imgs.shape[0]} frames)")



if __name__ == "__main__":
    main()
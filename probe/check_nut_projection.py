"""Sanity-check the world->pixel projection by overlaying the SquareNut's
ground-truth world-frame position as a cross on the agentview camera, for
every state in a robomimic HDF5.

If this video looks right (cross sits on the nut throughout), the
camera intrinsic/extrinsic + y-flip logic is correct and we can trust
the same projection inside the probe eval pipeline.

Example:
    python -m probe.check_nut_projection \\
        --dataset_path datasets/square/ph/low_dim_v15.hdf5 \\
        --task square \\
        --img_size 256 \\
        --n_demos 3 \\
        --every 1
"""

import argparse
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)

from envs.robomimic.robosuite import (
    make_env,
    _render_cams,
    _configure_for_dataset,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_sim(env):
    """Robomimic wraps robosuite; the MuJoCo sim sits one level deeper."""
    for path in ("sim", "env.sim", "base_env.sim"):
        obj = env
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find MuJoCo sim object on env")


def world_to_pixel(sim, point_xyz, camera_name, H, W, y_flip=True):
    K = get_camera_intrinsic_matrix(sim, camera_name, H, W)
    T = get_camera_extrinsic_matrix(sim, camera_name)
    p_world = np.append(point_xyz, 1.0)
    p_cam = T @ p_world
    if p_cam[2] <= 0:
        return None
    p_pix = K @ p_cam[:3]
    u, v = p_pix[0] / p_pix[2], p_pix[1] / p_pix[2]
    if y_flip:
        v = H - 1 - v
    return int(round(u)), int(round(v))


def draw_cross(img, uv, color=(255, 0, 0), size=18, thickness=2):
    if uv is None:
        return img
    h, w = img.shape[:2]
    if not (0 <= uv[0] < w and 0 <= uv[1] < h):
        return img
    cv2.drawMarker(
        img, uv, color=color, markerType=cv2.MARKER_CROSS,
        markerSize=size, thickness=thickness, line_type=cv2.LINE_AA,
    )
    return img


def get_nut_xyz(obs_dict):
    """Robomimic exposes the nut world-frame position either as a top-level
    key or as `object[7:10]` per the user's index table."""
    if "SquareNut_pos" in obs_dict:
        return np.asarray(obs_dict["SquareNut_pos"], dtype=np.float32)
    if "object" in obs_dict:
        return np.asarray(obs_dict["object"], dtype=np.float32)[7:10]
    raise RuntimeError(
        f"No nut-position key in obs_dict; have {sorted(obs_dict.keys())}"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--task", default="square")
    p.add_argument("--cam_key", default="agentview_image")
    p.add_argument("--img_size", type=int, default=256, help="render H=W")
    p.add_argument("--n_demos", type=int, default=2,
                   help="how many demos to render")
    p.add_argument("--every", type=int, default=1,
                   help="step stride (1 = every frame)")
    p.add_argument("--output_dir", default="nut_overlay_videos")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument(
        "--no_yflip", action="store_true",
        help="Disable the v = H-1-v line. Toggle this if the cross looks "
             "mirrored vertically.",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _configure_for_dataset(args.dataset_path)

    env = make_env(args.task, args.dataset_path)
    sim = _get_sim(env)
    cam_name = args.cam_key.replace("_image", "")
    H = W = args.img_size
    y_flip = not args.no_yflip
    print(f"camera={cam_name} render={H}x{W} y_flip={y_flip}")

    with h5py.File(args.dataset_path, "r") as f:
        all_demos = list(f["data"].keys())
        demos = all_demos[: args.n_demos]

        for di, demo_name in enumerate(demos):
            states = f[f"data/{demo_name}/states"][()]
            frames = []
            for si, s in enumerate(states[::args.every]):
                obs = env.reset_to({"states": s})
                nut_xyz = get_nut_xyz(obs)

                raw = _render_cams(env, [args.cam_key], height=H, width=W)[0]
                frame = raw.copy()
                uv = world_to_pixel(sim, nut_xyz, cam_name, H, W, y_flip=y_flip)
                draw_cross(frame, uv, color=(255, 0, 0))
                cv2.putText(
                    frame,
                    f"d={di} t={si} xyz=[{nut_xyz[0]:+.3f},"
                    f"{nut_xyz[1]:+.3f},{nut_xyz[2]:+.3f}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                frames.append(frame)

            out_path = os.path.join(args.output_dir, f"demo_{di:03d}.mp4")
            imageio.mimwrite(out_path, np.stack(frames), fps=args.fps, quality=8)
            print(f"demo {di} ({demo_name}): {len(frames)} frames -> {out_path}")

    print(f"done. videos in {args.output_dir}")


if __name__ == "__main__":
    main()

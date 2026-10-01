"""Re-execute recorded actions in a freshly-built robomimic env and save MP4s.

Builds the env from the dataset's `env_args`, resets, then steps through the
recorded actions. The longhist dataset uses `control_delta=false` (absolute
end-effector targets), so the trajectory should track the demonstration
closely even without state-snapping.

Note the env reset is whatever robomimic does by default (random) — it is not
the same initial state the original demo was collected from. The action
trajectory is correct; the *starting pose* may differ slightly from the
recorded demo.

Usage:
    MUJOCO_GL=egl python replay_longhist_in_sim.py \
        --dataset /home/ur10/Downloads/longhistsquare100/demos.hdf5 \
        --output_dir replays_sim --demos 0 1 --camera agentview --resolution 256
"""
import argparse
import os

import h5py
import imageio
import numpy as np

# Use the patched env factory from the pomdps repo so robosuite 1.5+ accepts
# the legacy controller config.
from envs.robomimic.robosuite import make_env, _render, _LDP_LONGHIST_RESET_STATE


def _infer_grasp_xy(actions):
    """Demo's HANDLE xy ≈ action xy at the first gripper-close. In absolute mode
    the eef hovers directly over the handle to grasp."""
    close_step = next((t for t in range(len(actions)) if actions[t, 6] > 0.5), None)
    if close_step is None:
        return None
    return actions[close_step, :2].astype(np.float64)


# With _LDP_LONGHIST_RESET_STATE's fixed nut quaternion, the SquareNut_handle_site
# sits at this world-frame offset from the body origin (measured via
# `check_nut_geometry.py`). To make the handle land at `grasp_xy`, place the
# body at `grasp_xy - HANDLE_OFFSET_XY`.
HANDLE_OFFSET_XY = np.array([0.0506, -0.0188])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", default="replays_sim")
    ap.add_argument("--demos", type=int, nargs="*", default=None)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--task", default="square")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building env...")
    env = make_env(args.task, args.dataset)

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(f["data"].keys(),
                           key=lambda k: int(k.split("_")[-1]))
        if args.demos is not None:
            demo_keys = [f"demo_{i}" for i in args.demos]

        for dk in demo_keys:
            actions = f[f"data/{dk}/actions"][()]
            print(f"\n{dk}: {actions.shape[0]} steps")

            # The dataset has no `states`, so default env.reset() puts the nut
            # wherever robomimic's RNG picks. Snap the handle to the demo's
            # grasp xy: place the body at grasp_xy - HANDLE_OFFSET_XY so the
            # handle site (offset by HANDLE_OFFSET_XY from body) lands on the
            # recorded grasp point.
            grasp_xy = _infer_grasp_xy(actions)
            if grasp_xy is None:
                print(f"  WARN: no gripper-close found in {dk}; using default reset")
                env.reset()
            else:
                body_xy = grasp_xy - HANDLE_OFFSET_XY
                state = _LDP_LONGHIST_RESET_STATE.copy()
                state[10:12] = body_xy
                env.reset_to({"states": state})
                print(f"  grasp xy = {grasp_xy.round(4).tolist()}, "
                      f"body xy = {body_xy.round(4).tolist()}")
            frames = []
            frames.append(_render(env, height=args.resolution, width=args.resolution,
                                  camera_name=args.camera))
            for t, a in enumerate(actions):
                env.step(a.astype(np.float64))
                frames.append(_render(env, height=args.resolution, width=args.resolution,
                                      camera_name=args.camera))
                if (t + 1) % 100 == 0:
                    print(f"  step {t + 1}/{actions.shape[0]}")

            out = os.path.join(args.output_dir, f"{dk}_{args.camera}_sim.mp4")
            imageio.mimsave(out, frames, fps=args.fps)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()

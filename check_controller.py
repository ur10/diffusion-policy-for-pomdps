"""Quick sanity check: did the legacy `control_delta` flag survive normalization
and reach the underlying robosuite controller? If `input_type` is "delta" for a
dataset whose actions were recorded as absolute targets, the controller will
silently misinterpret every action.

Usage:
    MUJOCO_GL=egl python check_controller.py --dataset /path/to/demos.hdf5
"""
import argparse
import json

import h5py

from envs.robomimic.robosuite import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", default="square")
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
    print("---- raw env_args.controller_configs ----")
    print(json.dumps(env_args["env_kwargs"]["controller_configs"], indent=2))

    env = make_env(args.task, args.dataset)

    # robomimic EnvRobosuite -> .env is the raw robosuite env
    rs = env.env
    robot = rs.robots[0]

    # robosuite >= 1.5: composite_controller.part_controllers["right"]
    # robosuite <  1.5: robot.controller
    if hasattr(robot, "composite_controller"):
        parts = robot.composite_controller.part_controllers
        print("\n---- composite_controller.part_controllers ----")
        for name, ctrl in parts.items():
            print(f"  [{name}] {type(ctrl).__name__}")
            print(f"     input_type     = {getattr(ctrl, 'input_type', '<missing>')!r}")
            print(f"     use_delta      = {getattr(ctrl, 'use_delta',  '<missing>')!r}")
            print(f"     control_dim    = {getattr(ctrl, 'control_dim','<missing>')!r}")
            print(f"     damping_ratio  = {getattr(ctrl, 'damping_ratio','<missing>')!r}")
            print(f"     kp             = {getattr(ctrl, 'kp',         '<missing>')!r}")
    else:
        ctrl = robot.controller
        print("\n---- robot.controller ----")
        print(f"  {type(ctrl).__name__}")
        print(f"  use_delta = {getattr(ctrl, 'use_delta', '<missing>')!r}")
        print(f"  kp        = {getattr(ctrl, 'kp', '<missing>')!r}")


if __name__ == "__main__":
    main()

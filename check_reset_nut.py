"""Verify whether _LDP_LONGHIST_RESET_STATE[10:12] really is the nut x/y, and
where the recorded demo's actions are aiming during the grasp phase. If the
recorded grasp xy != reset nut xy, then any replay (or eval that depends on
the nut being where the demo expects) will miss the nut.
"""
import argparse
import numpy as np
import h5py

from envs.robomimic.robosuite import (
    make_env,
    _LDP_LONGHIST_RESET_STATE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--task", default="square")
    args = ap.parse_args()

    env = make_env(args.task, args.dataset)
    rs = env.env

    # ---- 1) Confirm reset_state[10:12] maps to the nut body
    state = _LDP_LONGHIST_RESET_STATE.copy()
    target_xy = np.array([0.123, -0.077])     # arbitrary distinctive value
    state[10:12] = target_xy
    env.reset_to({"states": state})
    nut_world = np.array(rs.sim.data.body_xpos[rs.obj_body_id["SquareNut"]])
    print(f"set state[10:12] = {target_xy}")
    print(f"resulting nut world pos = {nut_world.round(4)}")
    print(f"  delta vs requested xy = {(nut_world[:2] - target_xy).round(4)}")
    print()

    # ---- 2) Inspect recorded demo grasp phase
    with h5py.File(args.dataset, "r") as f:
        d = f[f"data/{args.demo}"]
        actions = d["actions"][()]
        eef = d["obs/robot0_eef_pos"][()]
        grip = d["obs/robot0_gripper_qpos"][()]

    # Find first big gripper close: action[6] going to +1
    close_step = next((t for t in range(len(actions)) if actions[t, 6] > 0.5), None)
    print(f"{args.demo}: first gripper-close action at t={close_step}")
    if close_step is not None:
        print(f"  action[t]   = {actions[close_step].round(4).tolist()}")
        print(f"  eef pos[t]  = {eef[close_step].round(4).tolist()}")
        # Earliest descent xy is a good proxy for nut xy
        # (the absolute target lines up with where the nut is sitting)
        print(f"  action xy at t-5 = {actions[max(close_step-5, 0), :2].round(4)}")
        print(f"  action xy at t   = {actions[close_step, :2].round(4)}")

    # ---- 3) Compare to wrapper's randomized reset range
    print(f"\n_MultiStageSquareWrapper randomizes nut xy in:")
    print(f"  [-0.2, -0.2] to [0.0, 0.2]   (LDP convention)")
    print(f"  recorded grasp xy lies in this range? {actions[close_step, 0] >= -0.2 and actions[close_step, 0] <= 0.0 and actions[close_step, 1] >= -0.2 and actions[close_step, 1] <= 0.2}")

if __name__ == "__main__":
    main()

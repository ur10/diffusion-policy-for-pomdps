"""Step the env with a recorded absolute-target action and see if the eef
actually moves toward that target. If yes, the controller is interpreting the
action in the same frame the demo recorded. If no, we have a frame mismatch
(world vs base most likely)."""
import argparse
import numpy as np

from envs.robomimic.robosuite import make_env, _normalize_actions  # noqa
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", default="square")
    args = ap.parse_args()

    env = make_env(args.task, args.dataset)

    with h5py.File(args.dataset, "r") as f:
        # Grab the first recorded action of the first demo
        d = f["data/demo_0"]
        action0 = d["actions"][0].astype(np.float64)
        recorded_eef0 = d["obs/robot0_eef_pos"][0]

    rs = env.env  # underlying robosuite env
    robot = rs.robots[0]
    ctrl = robot.composite_controller.part_controllers["right"]
    print(f"controller: {type(ctrl).__name__}")
    print(f"  input_type      = {ctrl.input_type}")
    print(f"  input_ref_frame = {getattr(ctrl, 'input_ref_frame', '<missing>')}")
    print(f"  ref_pos (init)  = {ctrl.ref_pos}")  # eef pos in input_ref_frame
    base_pos = robot.base_pos
    print(f"  robot base_pos (world) = {base_pos}")

    obs = env.reset()
    eef_world_before = obs["robot0_eef_pos"]
    print(f"\neef world BEFORE step      = {eef_world_before}")
    print(f"recorded eef at demo step 0 = {recorded_eef0}")
    print(f"action[0:3]                 = {action0[:3]}")

    # Step many times so the controller can converge to the target
    for _ in range(20):
        obs, _, _, _ = env.step(action0)
    eef_world_after = obs["robot0_eef_pos"]
    print(f"\neef world AFTER 20 steps   = {eef_world_after}")
    print(f"target if action is WORLD  : err = {np.linalg.norm(eef_world_after - action0[:3]):.4f} m")
    print(f"target if action is BASE   : err = "
          f"{np.linalg.norm(eef_world_after - (action0[:3] + base_pos)):.4f} m")


if __name__ == "__main__":
    main()

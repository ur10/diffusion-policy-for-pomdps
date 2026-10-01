"""Spawn the SquareNut at a known body xy and inspect its geom layout to find
the offset between the body origin and the grasp handle. The longhist demos
record the eef hovering over the handle, but `state[10:12]` controls the body
origin -- if those differ by a fixed offset, replays mis-place the nut.
"""
import argparse
import numpy as np

from envs.robomimic.robosuite import make_env, _LDP_LONGHIST_RESET_STATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", default="square")
    args = ap.parse_args()

    env = make_env(args.task, args.dataset)
    rs = env.env

    # spawn the nut at a known body xy
    body_xy = np.array([0.0, 0.0])
    state = _LDP_LONGHIST_RESET_STATE.copy()
    state[10:12] = body_xy
    env.reset_to({"states": state})

    body_id = rs.obj_body_id["SquareNut"]
    body_xpos = np.array(rs.sim.data.body_xpos[body_id])
    print(f"requested body xy = {body_xy}")
    print(f"actual body xpos  = {body_xpos.round(4)}")

    # robosuite stores grasp sites for nut/peg objects.
    # Look for a site with 'handle' / 'grip' / 'SquareNut' in the name
    site_names = rs.sim.model.site_names
    print(f"\nsites containing 'nut' or 'handle' or 'grip':")
    interesting = [s for s in site_names
                   if any(t in s.lower() for t in ("nut", "handle", "grip"))]
    for s in interesting:
        sid = rs.sim.model.site_name2id(s)
        spos = np.array(rs.sim.data.site_xpos[sid])
        print(f"  {s:30s} world = {spos.round(4).tolist()}  "
              f"offset from body = {(spos - body_xpos).round(4).tolist()}")

    # SquareNut also has named geoms; print those too
    geom_names = rs.sim.model.geom_names
    print(f"\ngeoms containing 'nut' / 'square' / 'handle':")
    for g in geom_names:
        if g and any(t in g.lower() for t in ("nut", "square", "handle")):
            gid = rs.sim.model.geom_name2id(g)
            gpos = np.array(rs.sim.data.geom_xpos[gid])
            print(f"  {g:35s} world = {gpos.round(4).tolist()}  "
                  f"offset from body = {(gpos - body_xpos).round(4).tolist()}")


if __name__ == "__main__":
    main()

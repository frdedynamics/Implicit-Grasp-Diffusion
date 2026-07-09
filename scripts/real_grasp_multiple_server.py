#!/usr/bin/env python3
"""
igd_inference_server.py -- runs inside your IGD conda env (NOT the ROS env).

Receives raw depth frames + camera intrinsics + extrinsics (T_cam_task) from
tsdf_grasp_node.py over ZMQ, rebuilds the TSDF with the repo's own
TSDFVolume class (identical to how simulation builds it), runs the IGD
detector, and replies with grasp poses in the task frame + the fused cloud.

Run from the repo root so the imports resolve:

    conda activate igd_blackwell
    cd ~/Implicit-Grasp-Diffusion
    python igd_inference_server.py --model data/models/IGD_pile.pt

pip install pyzmq in the conda env if missing.

!!! ADAPT THE THREE SPOTS MARKED  # >>> CHECK <<<  to your fork. They mirror
    what scripts/sim_grasp_multiple.py does -- open that file side by side
    and copy the exact detector constructor / call you use in simulation.
"""

import argparse
import pickle
import traceback

import numpy as np
import zmq

# >>> CHECK <<< import paths: match whatever sim_grasp_multiple.py imports.
# This fork keeps the detector under src/igd/. Confirmed:
#   src/igd/detection_implicit.py -> VGNImplicit
from igd.detection_implicit import VGNImplicit
# >>> VERIFY <<< these two: some forks keep perception/transform under `vgn`,
# others moved everything under `igd`. Check the actual folders under src/ and
# adjust. Try `igd.` first; if ImportError, use `vgn.`.
from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform


def build_state(frames, size, resolution):
    """Rebuild the TSDF exactly like ClutterRemovalSim.acquire_tsdf().

    Mirrors simulation.py:
      - network TSDF at `resolution` (40)
      - point cloud from a SEPARATE 120-res TSDF
      - crop to self.lower..self.upper, which for size=0.30, table_height=0.05:
            lower = [0.02, 0.02, 0.055]      (2cm inset in x/y, floor at h+0.005)
            upper = [0.28, 0.28, 0.30]
    """
    import open3d as o3d

    tsdf = TSDFVolume(size, resolution)
    high_res_tsdf = TSDFVolume(size, 120)

    for f in frames:
        intrinsic = CameraIntrinsic(f["width"], f["height"],
                                    f["fx"], f["fy"], f["cx"], f["cy"])
        # integrate() wants a Transform that maps task-frame points -> camera
        # frame (world->camera extrinsic). The ROS node computes exactly this as
        # T_cam_task = lookup_transform(camera_frame, task_frame).
        extrinsic = Transform.from_matrix(np.asarray(f["T_cam_task"], np.float64))
        depth = np.asarray(f["depth"], np.float32)
        tsdf.integrate(depth, intrinsic, extrinsic)
        high_res_tsdf.integrate(depth, intrinsic, extrinsic)

    # crop bounds exactly as place_table() defines them (inset + floor at h+0.005)
    inset = 0.02
    floor = 0.05 + 0.005          # table_height + 0.005
    lower = np.array([inset, inset, floor])
    upper = np.array([size - inset, size - inset, size])
    pc = high_res_tsdf.get_cloud()
    pc = pc.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))

    state = argparse.Namespace(tsdf=tsdf, pc=pc)
    return state


def grasp_to_dict(grasp, score):
    """vgn.grasp.Grasp -> plain dict (pose is T_task_grasp)."""
    pose = grasp.pose
    return {
        "position": np.asarray(pose.translation, np.float64).tolist(),
        "quaternion_xyzw": np.asarray(pose.rotation.as_quat(), np.float64).tolist(),
        "width": float(grasp.width),
        "score": float(score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to IGD checkpoint (.pt)")
    ap.add_argument("--model-type", default="igd",
                    help=">>> CHECK <<< same --type string you pass to "
                         "sim_grasp_multiple.py")
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--qual-th", type=float, default=0.9)  # matches sim default
    ap.add_argument("--bind", default="tcp://127.0.0.1:5555")
    args = ap.parse_args()

    # Matches VGNImplicit.__init__ in src/igd/detection_implicit.py, with the
    # flags from the sim command: --force --best --type igd (qual-th 0.9 default).
    # visualize MUST stay False here: the visualize=True path returns a 4-tuple
    # (grasps, scores, timing, visual_mesh) and we don't want a mesh over ZMQ.
    detector = VGNImplicit(
        args.model,
        args.model_type,        # "igd"
        best=True,              # --best
        force_detection=True,   # --force
        qual_th=args.qual_th,   # 0.9 to match sim
        out_th=0.5,
        visualize=False,
        resolution=args.resolution,
    )
    print(f"[igd-server] model loaded: {args.model} (type={args.model_type})")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[igd-server] listening on {args.bind}")

    while True:
        req = pickle.loads(sock.recv())
        try:
            if req.get("cmd") != "predict":
                raise ValueError(f"unknown cmd: {req.get('cmd')}")

            frames = req["frames"]
            size = float(req.get("size", 0.30))
            print(f"[igd-server] predict: {len(frames)} frame(s), size={size}")

            state = build_state(frames, size, args.resolution)

            # Matches clutter_removal.py non-visualize path:
            #   grasps, scores, timings["planning"] = grasp_plan_fn(state, last_trial)
            # detector built with visualize=False, so it returns a 3-tuple.
            # last_trial=None -> fresh scene, no retry context.
            grasps, scores, toc = detector(state, None)

            reply = {
                "grasps": [grasp_to_dict(g, s) for g, s in zip(grasps, scores)],
                "cloud": np.asarray(state.pc.points, np.float32),
                "toc": float(toc),
                "error": None,
            }
            print(f"[igd-server] -> {len(grasps)} grasps in {toc:.3f}s"
                  + (f", best score {max(scores):.3f}" if len(scores) else ""))
        except Exception:
            reply = {"grasps": [], "cloud": None, "toc": None,
                     "error": traceback.format_exc()}
            print(reply["error"])
        sock.send(pickle.dumps(reply, protocol=4))


if __name__ == "__main__":
    main()
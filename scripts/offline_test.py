#!/usr/bin/env python3
"""
offline_test.py -- run entirely inside the IGD conda env. No ROS, no robot.

Loads one saved real depth frame + intrinsics + extrinsic, rebuilds the state
exactly like the sim, runs the IGD detector, and opens an Open3D window with
the scene cloud + the predicted grasps drawn as gripper wireframes.

This is the single most important check before touching the robot: if the
grasps point sensibly INTO objects here, the whole perception path is correct.

Usage:
    conda activate igd_blackwell
    cd ~/Implicit-Grasp-Diffusion
    python offline_test.py \
        --depth /tmp/depth.npy \
        --fx 640.5 --fy 640.5 --cx 641.2 --cy 366.9 \
        --width 1280 --height 720 \
        --extrinsic /tmp/extrinsic.npy \
        --model ./data/models/IGD_pile.pt --type igd

--extrinsic is a 4x4 .npy mapping TASK-frame points -> CAMERA frame
(i.e. lookup_transform(camera_color_optical_frame, task)). See make_extrinsic().
"""
import argparse
import numpy as np

from igd.detection_implicit import VGNImplicit
from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform
import open3d as o3d


SIZE = 0.30


def build_state(depth_m, K, extrinsic_mat, resolution=40):
    tsdf = TSDFVolume(SIZE, resolution)
    high = TSDFVolume(SIZE, 120)
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    extr = Transform.from_matrix(extrinsic_mat)
    tsdf.integrate(depth_m, intr, extr)
    high.integrate(depth_m, intr, extr)

    inset, floor = 0.02, 0.055
    lower = np.array([inset, inset, floor])
    upper = np.array([SIZE - inset, SIZE - inset, SIZE])
    pc = high.get_cloud().crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))
    return argparse.Namespace(tsdf=tsdf, pc=pc), pc


def gripper_lineset(grasp, color):
    """Draw a simple 2-finger gripper wireframe at grasp.pose (Franka-ish)."""
    w = float(grasp.width)
    d = 0.05  # finger depth
    pts = np.array([
        [0, 0, -d], [0, 0, 0],          # wrist stem, z = approach
        [-w / 2, 0, 0], [w / 2, 0, 0],  # bar
        [-w / 2, 0, d], [w / 2, 0, d],  # fingers forward
    ])
    T = grasp.pose.as_matrix()
    pts = (T[:3, :3] @ pts.T).T + T[:3, 3]
    lines = [[0, 1], [2, 3], [2, 4], [3, 5]]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(pts),
        o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True, help=".npy depth image")
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--extrinsic", required=True, help="4x4 .npy, task->camera")
    ap.add_argument("--model", required=True)
    ap.add_argument("--type", default="igd")
    ap.add_argument("--depth-scale", type=float, default=1000.0,
                    help="divide raw depth by this to get metres "
                         "(1000 for uint16 mm, 1.0 if already metres)")
    args = ap.parse_args()

    depth = np.load(args.depth).astype(np.float32)
    if depth.max() > 100:           # clearly millimetres
        depth = depth / args.depth_scale
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth > 1.2] = 0.0

    K = dict(width=args.width, height=args.height,
             fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    extr = np.load(args.extrinsic).astype(np.float64)

    state, pc = build_state(depth, K, extr)
    print(f"cloud points after crop: {len(pc.points)}")
    if len(pc.points) < 100:
        print("WARNING: almost empty cloud. Check extrinsic / task frame / "
              "depth scale before trusting anything below.")

    detector = VGNImplicit(args.model, args.type, best=True,
                           force_detection=True, qual_th=0.9,
                           out_th=0.5, visualize=False, resolution=40)
    grasps, scores, toc = detector(state, None)
    print(f"{len(grasps)} grasps in {toc:.3f}s")
    for i, (g, s) in enumerate(zip(grasps, scores)):
        p = g.pose.translation
        print(f"  [{i}] score={s:.3f} pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
              f"width={g.width:.3f}")

    # visualize: cloud + workspace box + grasps (best in green, rest in gray)
    geoms = [pc]
    box = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [SIZE, SIZE, SIZE])
    box.color = (0.1, 0.8, 0.1)
    geoms.append(box)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    if len(scores):
        best = int(np.argmax(scores))
        for i, g in enumerate(grasps):
            geoms.append(gripper_lineset(
                g, [0.1, 0.8, 0.1] if i == best else [0.6, 0.6, 0.6]))
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
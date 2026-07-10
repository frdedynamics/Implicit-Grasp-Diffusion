#!/usr/bin/env python3
"""
check_table_plane.py -- run in the IGD conda env.

Rebuilds the cloud from a saved depth frame + extrinsic (same as offline_test,
but WITHOUT cropping), fits a plane to the dominant surface (the table), and
reports:
  - the table normal in TASK coordinates
  - the tilt of that normal away from the task z-axis (degrees)
  - the mean table height (task z)

If tilt is more than ~1-2 deg, your task frame is not level with the table.
The printed roll/pitch tell you how to correct the static_transform_publisher
that defines the task frame.

Usage:
    python check_table_plane.py \
        --depth /tmp/depth.npy --extrinsic /tmp/extrinsic.npy \
        --fx 640 --fy 640 --cx 640 --cy 360 --width 1280 --height 720
"""
import argparse
import numpy as np
import open3d as o3d

from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform

SIZE = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True)
    ap.add_argument("--extrinsic", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    args = ap.parse_args()

    depth = np.load(args.depth).astype(np.float32)
    if depth.max() > 100:
        depth /= 1000.0
    depth = np.nan_to_num(depth)
    depth[depth > 1.2] = 0.0

    high = TSDFVolume(SIZE, 120)
    intr = CameraIntrinsic(args.width, args.height,
                           args.fx, args.fy, args.cx, args.cy)
    extr = Transform.from_matrix(np.load(args.extrinsic).astype(np.float64))
    high.integrate(depth, intr, extr)
    pc = high.get_cloud()

    # keep only points inside the cube (uncropped in z so table is included)
    pts = np.asarray(pc.points)
    m = np.all((pts >= 0) & (pts <= SIZE), axis=1)
    pc = pc.select_by_index(np.where(m)[0])
    print(f"points in cube: {len(pc.points)}")

    # RANSAC plane fit -> the table is the dominant plane
    plane, inliers = pc.segment_plane(distance_threshold=0.004,
                                      ransac_n=3, num_iterations=1000)
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)
    if n[2] < 0:          # point normal "up" in +z
        n, d = -n, -d
    n /= np.linalg.norm(n)

    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    inlier_pts = np.asarray(pc.points)[inliers]
    height = float(np.mean(inlier_pts[:, 2]))

    print(f"\ntable normal (task frame): [{n[0]:+.4f} {n[1]:+.4f} {n[2]:+.4f}]")
    print(f"tilt from task z-axis    : {tilt:.2f} deg")
    print(f"table plane inliers      : {len(inliers)} "
          f"({100*len(inliers)/len(pc.points):.0f}% of cube points)")
    print(f"mean table height (task z): {height:.4f} m")

    # roll/pitch that would make task-z align with the measured table normal.
    # These are the CORRECTIONS to fold into your task static transform.
    roll = np.degrees(np.arctan2(n[1], n[2]))
    pitch = np.degrees(np.arctan2(-n[0], np.hypot(n[1], n[2])))
    print(f"\nsuggested task-frame correction:")
    print(f"  roll  (about x): {roll:+.2f} deg")
    print(f"  pitch (about y): {pitch:+.2f} deg")

    if tilt < 1.0:
        print("\n=> Table is essentially level. Tilt is not your problem.")
    else:
        print(f"\n=> Table is tilted {tilt:.1f} deg in task frame. Re-publish the")
        print("   task static transform with the roll/pitch above (and set z so")
        print("   the table sits at ~0.05 in task coords), then re-capture.")


if __name__ == "__main__":
    main()
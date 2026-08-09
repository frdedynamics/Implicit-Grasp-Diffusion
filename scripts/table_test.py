#!/usr/bin/env python3
"""
Fits a plane to a fused point cloud (from build_state's pc) and reports how
many degrees its normal deviates from the task frame's Z axis. If task and
the real table are aligned (as RViz suggests), this should be close to 0.
Run this on the same view_pairs you feed into offline_test.py.
"""
import argparse
import numpy as np
import open3d as o3d

from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform

SIZE = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", nargs="+", required=True)
    ap.add_argument("--extrinsic", nargs="+", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    args = ap.parse_args()

    K = dict(width=args.width, height=args.height,
              fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])

    high = TSDFVolume(SIZE, 120)
    for depth_path, extr_path in zip(args.depth, args.extrinsic):
        depth = np.load(depth_path).astype(np.float32)
        if depth.max() > 100:
            depth = depth / args.depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 1.2] = 0.0
        extr_mat = np.load(extr_path).astype(np.float64)
        extr = Transform.from_matrix(extr_mat)
        high.integrate(depth, intr, extr)

    pc = high.get_cloud()
    print(f"total fused points: {len(pc.points)}")

    # crude table isolation: take the lowest 25% of points by Z, which should
    # be dominated by the table surface rather than the box sides
    pts = np.asarray(pc.points)
    z_thresh = np.percentile(pts[:, 2], 25)
    table_pts = pts[pts[:, 2] <= z_thresh]
    print(f"points used for table plane fit (lowest 25% by Z): {len(table_pts)}")

    table_pc = o3d.geometry.PointCloud()
    table_pc.points = o3d.utility.Vector3dVector(table_pts)
    plane_model, inliers = table_pc.segment_plane(
        distance_threshold=0.004, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    # angle between fitted normal and task's +Z axis
    z_axis = np.array([0, 0, 1])
    cos_angle = np.clip(np.abs(np.dot(normal, z_axis)), -1, 1)
    tilt_deg = np.degrees(np.arccos(cos_angle))

    print(f"\nFitted table plane normal (task frame): {normal}")
    print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"Inlier count: {len(inliers)} / {len(table_pts)}")
    print(f"\n>>> Table tilt relative to task Z-axis: {tilt_deg:.2f} degrees <<<")
    print("If RViz shows task and table aligned, this should be small (<~3-5 deg).")
    print("A large number here (10+ deg) confirms a real mismatch between the")
    print("TF-defined task frame and what's actually being fused.")


if __name__ == "__main__":
    main()
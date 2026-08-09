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

--per-view-debug: before fusing, opens an Open3D window showing each view's
raw (unfused) point cloud in its own color, all in the same task frame. This
lets you visually check whether the extrinsics actually agree with each other
before you spend any time debugging the detector. Well-aligned views should
look like one coherent surface; misaligned views show doubled/ghosted edges.
"""
import argparse
import numpy as np

from igd.detection_implicit import VGNImplicit
from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform
import open3d as o3d


SIZE = 0.30

# distinct, easy-to-tell-apart colors for up to 8 per-view debug clouds
DEBUG_COLORS = [
    [0.90, 0.10, 0.10],  # red
    [0.10, 0.55, 0.90],  # blue
    [0.10, 0.80, 0.10],  # green
    [0.95, 0.65, 0.10],  # orange
    [0.65, 0.10, 0.90],  # purple
    [0.10, 0.85, 0.85],  # cyan
    [0.90, 0.10, 0.65],  # pink
    [0.75, 0.75, 0.10],  # olive
]


def build_state(depth_extrinsic_pairs, K, resolution=40):
    """
    depth_extrinsic_pairs: list of (depth_m, extrinsic_mat) tuples,
    one per camera view. All extrinsics must map the SAME task frame
    -> that view's camera frame.
    """
    tsdf = TSDFVolume(SIZE, resolution)
    high = TSDFVolume(SIZE, 120)
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])

    for depth_m, extrinsic_mat in depth_extrinsic_pairs:
        extr = Transform.from_matrix(extrinsic_mat)
        tsdf.integrate(depth_m, intr, extr)
        high.integrate(depth_m, intr, extr)

    inset, floor = 0.02, 0.055
    lower = np.array([inset, inset, floor])
    upper = np.array([SIZE - inset, SIZE - inset, SIZE])
    pc = high.get_cloud().crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    labels = np.array(pc.cluster_dbscan(eps=0.006, min_points=15))
    if labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        pc = pc.select_by_index(np.where(labels == largest)[0])
    return argparse.Namespace(tsdf=tsdf, pc=pc), pc


def per_view_clouds(depth_extrinsic_pairs, K, resolution=120):
    """
    Build one small TSDF PER VIEW (no fusion across views) and return the
    resulting point clouds, each tagged with a debug color, still expressed
    in the shared task frame. Purely for visual alignment-checking; not used
    by the detector.
    """
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    clouds = []
    for i, (depth_m, extrinsic_mat) in enumerate(depth_extrinsic_pairs):
        view_tsdf = TSDFVolume(SIZE, resolution)
        extr = Transform.from_matrix(extrinsic_mat)
        view_tsdf.integrate(depth_m, intr, extr)
        pc = view_tsdf.get_cloud()
        color = DEBUG_COLORS[i % len(DEBUG_COLORS)]
        pc.paint_uniform_color(color)
        clouds.append(pc)
        print(f"  view [{i}] color={color} points={len(pc.points)}")
    return clouds


def remove_edge_flying_pixels(depth_m, grad_threshold=0.02):
    """
    Zero out depth pixels sitting on a sharp local depth discontinuity --
    these are almost always flying-pixel artifacts, not real surface.
    grad_threshold is in metres; tune based on your scene (start ~0.01-0.03).
    """
    gy, gx = np.gradient(depth_m)
    grad_mag = np.sqrt(gx**2 + gy**2)
    cleaned = depth_m.copy()
    cleaned[grad_mag > grad_threshold] = 0.0
    return cleaned


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
    ap.add_argument("--depth", nargs="+", required=True)
    ap.add_argument("--extrinsic", nargs="+", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--type", default="igd")
    ap.add_argument("--qual-th", type=float, default=0.0,
                    help="lower (e.g. 0.1) to inspect all candidates")
    ap.add_argument("--depth-scale", type=float, default=1000.0,
                    help="divide raw depth by this to get metres "
                         "(1000 for uint16 mm, 1.0 if already metres)")
    ap.add_argument("--per-view-debug", action="store_true",
                    help="show each view's unfused point cloud in its own "
                         "color (in the shared task frame) before running "
                         "detection, to visually check extrinsic alignment")
    args = ap.parse_args()

    if len(args.depth) != len(args.extrinsic):
        raise ValueError(f"Got {len(args.depth)} depth files but {len(args.extrinsic)} extrinsics — need one extrinsic per depth image, in matching order.")

    K = dict(width=args.width, height=args.height,
             fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    view_pairs = []
    for depth_path, extr_path in zip(args.depth, args.extrinsic):
        depth = np.load(depth_path).astype(np.float32)
        if depth.max() > 100:
            depth = depth / args.depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 1.2] = 0.0
        depth = remove_edge_flying_pixels(depth, grad_threshold=0.005)  # 0.02
        extr = np.load(extr_path).astype(np.float64)
        print(f"File: {extr_path} | Translation: {extr[:3, 3]}")
        view_pairs.append((depth, extr))

    if args.per_view_debug:
        print(f"--per-view-debug: building {len(view_pairs)} unfused per-view clouds...")
        debug_clouds = per_view_clouds(view_pairs, K)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        box = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [SIZE, SIZE, SIZE])
        box.color = (0.1, 0.1, 0.1)
        print("Opening per-view debug window. Well-aligned views should look "
              "like one coherent surface; doubled/ghosted edges mean an "
              "extrinsic (or timing) mismatch between views.")
        o3d.visualization.draw_geometries(debug_clouds + [frame, box])

    state, pc = build_state(view_pairs, K)
    print(f"cloud points after crop: {len(pc.points)}")
    if len(pc.points) < 100:
        print("WARNING: almost empty cloud. Check extrinsic / task frame / "
              "depth scale before trusting anything below.")

    detector = VGNImplicit(args.model, args.type, best=True,
                           force_detection=True, qual_th=args.qual_th,
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
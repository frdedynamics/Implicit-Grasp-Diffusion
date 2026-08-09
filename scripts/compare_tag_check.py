#!/usr/bin/env python3
"""
align_merge_tsdf.py -- Implicit-Grasp-Diffusion offline evaluation pipeline
with an ICP-based extrinsic correction step before TSDF fusion.

Why this exists
----------------
Your AprilTag cross-check (compare_tag_check.py) shows a consistent ~2cm
offset between views, coming from a hand-eye calibration error. Rather than
re-deriving the calibration, this script treats each view's own point cloud
(reconstructed from its depth image + its own extrinsic) as ground truth for
*that view's local geometry*, and finds a small rigid correction per view
that brings all views into global agreement, using the standard Open3D
multiway-registration recipe (pairwise ICP + pose-graph optimization).

The corrected extrinsics are then used for TSDF integration, exactly as in
your original script, so nothing downstream (VGNImplicit, grasp detection,
visualization) needs to change.

IMPORTANT CAVEATS
-----------------
- This assumes the *scene geometry itself* (the object silhouette) is what
  should agree across views -- it doesn't know about your robot's kinematics,
  so it can't tell "the calibration is wrong" from "the object moved between
  captures". Keep the object static while capturing all views.
- ICP needs real geometric overlap between views to converge correctly. It
  works well when views converge on a common object from different angles
  (which is your case), but can silently converge to a wrong local minimum
  if two views barely overlap or the object is nearly symmetric. Always
  check the printed per-view fix magnitudes (see below) -- if one view's
  fix is much larger than a couple cm, treat its output with suspicion
  rather than trusting it blindly.
- This is a per-capture workaround, not a permanent calibration fix. If the
  ~2cm bias is constant across captures, it's worth still chasing down in
  your hand-eye calibration when you have time.

Usage
-----
Same CLI as your original evaluate script, plus:
    --align                 enable the ICP correction step (default: on)
    --no-align               disable it (falls back to raw extrinsics, for
                              A/B comparison)
    --icp-coarse FLOAT        coarse ICP correspondence distance (m), default 0.03
    --icp-fine FLOAT           fine ICP correspondence distance (m),   default 0.008
"""

import argparse
import numpy as np
import open3d as o3d

from igd.detection_implicit import VGNImplicit
from igd.perception import TSDFVolume, CameraIntrinsic
from igd.utils.transform import Transform

SIZE = 0.30

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

# Constant transform that re-centers the [SIZE, SIZE, SIZE] TSDF cube on the
# task origin. Kept identical to your original script.
_HALF = SIZE / 2.0
T_TSDFCENTER_TO_TSDFORIGIN = np.eye(4)
T_TSDFCENTER_TO_TSDFORIGIN[:3, 3] = [-_HALF, -_HALF, -_HALF]


def remove_edge_flying_pixels(depth_m, grad_threshold=0.003):
    """Strips flying-pixel/edge artifacts around thin geometry."""
    gy, gx = np.gradient(depth_m)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    cleaned = depth_m.copy()
    cleaned[grad_mag > grad_threshold] = 0.0
    return cleaned


def load_views(depth_paths, extr_paths, depth_scale=1000.0):
    """Loads and cleans depth + (raw, uncorrected) extrinsic pairs."""
    view_pairs = []
    for depth_path, extr_path in zip(depth_paths, extr_paths):
        depth = np.load(depth_path).astype(np.float32)
        if depth.max() > 100:
            depth = depth / depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 1.2] = 0.0
        depth = remove_edge_flying_pixels(depth, grad_threshold=0.003)

        extr = np.load(extr_path).astype(np.float64)
        view_pairs.append((depth, extr))
    return view_pairs


def centered_extrinsics(view_pairs):
    """extrinsic_mat @ T_tsdfCenter_to_tsdfOrigin for every view (raw, pre-fix)."""
    return [extr @ T_TSDFCENTER_TO_TSDFORIGIN for _, extr in view_pairs]


def reconstruct_view_cloud(depth_m, intr, corrected_extrinsic, resolution=120):
    """Reconstructs a single-view point cloud (in the centered task frame)."""
    tsdf = TSDFVolume(SIZE, resolution)
    extr = Transform.from_matrix(corrected_extrinsic)
    tsdf.integrate(depth_m, intr, extr)
    return tsdf.get_cloud()


def pairwise_registration(source, target, max_corr_coarse, max_corr_fine):
    """Two-stage (coarse->fine) point-to-plane ICP between two clouds.

    Returns (transformation, information) where `transformation` maps
    points from `source`'s frame into `target`'s frame:
        p_target_frame = transformation @ p_source_frame
    """
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_corr_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_corr_fine, icp_fine.transformation)
    return icp_fine.transformation, information, icp_fine.fitness, icp_fine.inlier_rmse


def multiway_register(clouds, max_corr_coarse=0.03, max_corr_fine=0.008,
                       normal_radius=0.01, normal_max_nn=30):
    """
    Standard Open3D multiway registration: build a fully-connected pose
    graph from pairwise ICP, then globally optimize it.

    Returns a list of 4x4 "fix" transforms, one per view, each defined so
    that:
        p_fixed_i = fix[i] @ p_original_i
    where p_original_i are the points reconstructed from view i's own
    (uncorrected) extrinsic.
    """
    n = len(clouds)
    if n < 2:
        return [np.eye(4) for _ in range(n)]

    # Normals are required for point-to-plane ICP.
    for c in clouds:
        c.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    print("\nPairwise ICP fit report (lower inlier_rmse = better; "
          "fitness closer to 1.0 = more overlap used):")
    for source_id in range(n):
        for target_id in range(source_id + 1, n):
            transformation, information, fitness, rmse = pairwise_registration(
                clouds[source_id], clouds[target_id], max_corr_coarse, max_corr_fine)
            print(f"  view[{source_id}] -> view[{target_id}]: "
                  f"fitness={fitness:.3f} inlier_rmse={rmse * 1000:.2f}mm "
                  f"translation_correction={np.linalg.norm(transformation[:3, 3]) * 1000:.1f}mm")

            if target_id == source_id + 1:
                # Sequential ("odometry") edge -- also seeds that node's pose.
                node_pose = transformation @ pose_graph.nodes[source_id].pose
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(node_pose))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, transformation, information, uncertain=False))
            else:
                # Extra ("loop closure") edge -- helps the optimizer average
                # out per-pair ICP noise instead of trusting a single chain.
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id, target_id, transformation, information, uncertain=True))

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    # pose_graph.nodes[i].pose already maps p_original_i -> globally
    # consistent frame (i.e. exactly the "fix" transform we want), with
    # view 0 held fixed as the reference frame.
    fixes = [node.pose for node in pose_graph.nodes]

    print("\nFinal per-view correction magnitude (post pose-graph optimization):")
    for i, fix in enumerate(fixes):
        t = fix[:3, 3]
        print(f"  view[{i}]: translation_fix={np.linalg.norm(t) * 1000:.1f}mm "
              f"({t[0]*1000:+.1f}, {t[1]*1000:+.1f}, {t[2]*1000:+.1f})mm")
    return fixes


def build_state_aligned(view_pairs, K, resolution=80, high_resolution=120,
                         align=True, icp_coarse=0.03, icp_fine=0.008):
    """
    Like your original build_state, but first ICP-aligns the per-view
    clouds and folds the resulting correction into each view's extrinsic
    before integrating the shared TSDF.
    """
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    base_extrinsics = centered_extrinsics(view_pairs)  # extrinsic @ T_center, pre-fix

    if align and len(view_pairs) > 1:
        # Reconstruct each view alone (at a reasonably high resolution) to
        # get clean clouds for ICP.
        raw_clouds = [
            reconstruct_view_cloud(depth_m, intr, base_extrinsics[i], resolution=high_resolution)
            for i, (depth_m, _) in enumerate(view_pairs)
        ]
        fixes = multiway_register(raw_clouds, max_corr_coarse=icp_coarse, max_corr_fine=icp_fine)
    else:
        fixes = [np.eye(4) for _ in view_pairs]

    # Fold each fix into that view's extrinsic:
    #   p_fixed = fix_i @ inv(T_cam_taskc_i) @ p_cam
    # is equivalent to integrating with extrinsic' = T_cam_taskc_i @ inv(fix_i)
    final_extrinsics = [
        base_extrinsics[i] @ np.linalg.inv(fixes[i]) for i in range(len(view_pairs))
    ]

    tsdf = TSDFVolume(SIZE, resolution)
    high = TSDFVolume(SIZE, high_resolution)
    for (depth_m, _), extr_mat in zip(view_pairs, final_extrinsics):
        extr = Transform.from_matrix(extr_mat)
        tsdf.integrate(depth_m, intr, extr)
        high.integrate(depth_m, intr, extr)

    inset, floor = 0.02, 0.015
    lower = np.array([inset, inset, floor])
    upper = np.array([SIZE - inset, SIZE - inset, SIZE])
    pc = high.get_cloud().crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    labels = np.array(pc.cluster_dbscan(eps=0.006, min_points=15))
    if labels.max() >= 0:
        largest = np.bincount(labels[labels >= 0]).argmax()
        pc = pc.select_by_index(np.where(labels == largest)[0])

    return argparse.Namespace(tsdf=tsdf, pc=pc), pc, final_extrinsics


def per_view_clouds_debug(view_pairs, extrinsics, K, resolution=120):
    """Colored per-view clouds for visual sanity-checking, using whatever
    extrinsics are passed in (raw or corrected)."""
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    clouds = []
    for i, ((depth_m, _), extr_mat) in enumerate(zip(view_pairs, extrinsics)):
        view_tsdf = TSDFVolume(SIZE, resolution)
        extr = Transform.from_matrix(extr_mat)
        view_tsdf.integrate(depth_m, intr, extr)
        pc = view_tsdf.get_cloud()
        color = DEBUG_COLORS[i % len(DEBUG_COLORS)]
        pc.paint_uniform_color(color)
        clouds.append(pc)
        print(f"  view [{i}] color={color} points={len(pc.points)}")
    return clouds


def gripper_lineset(grasp, color):
    w = float(grasp.width)
    d = 0.05
    pts = np.array([
        [0, 0, -d], [0, 0, 0],
        [-w / 2, 0, 0], [w / 2, 0, 0],
        [-w / 2, 0, d], [w / 2, 0, d],
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
    ap.add_argument("--qual-th", type=float, default=0.0)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--per-view-debug", action="store_true")
    ap.add_argument("--align", dest="align", action="store_true", default=True,
                     help="ICP-align per-view clouds before TSDF fusion (default: on)")
    ap.add_argument("--no-align", dest="align", action="store_false",
                     help="Disable ICP alignment (use raw extrinsics as-is)")
    ap.add_argument("--icp-coarse", type=float, default=0.03,
                     help="Coarse ICP correspondence distance in meters")
    ap.add_argument("--icp-fine", type=float, default=0.008,
                     help="Fine ICP correspondence distance in meters")
    args = ap.parse_args()

    if len(args.depth) != len(args.extrinsic):
        raise ValueError("Mismatched pairs between input image sequences and extrinsic configurations.")

    K = dict(width=args.width, height=args.height,
             fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    view_pairs = load_views(args.depth, args.extrinsic, depth_scale=args.depth_scale)

    print(f"Aligning {len(view_pairs)} views before merging "
          f"({'ICP correction ON' if args.align else 'ICP correction OFF'})...")
    state, pc, final_extrinsics = build_state_aligned(
        view_pairs, K, resolution=80, high_resolution=120,
        align=args.align, icp_coarse=args.icp_coarse, icp_fine=args.icp_fine)

    print(f"\nCloud point density following filter step: {len(pc.points)}")
    if len(pc.points) < 100:
        print("WARNING: Insufficient structural cloud coordinates generated.")

    if args.per_view_debug:
        print(f"\n--per-view-debug: rendering {len(view_pairs)} corrected per-view clouds...")
        debug_clouds = per_view_clouds_debug(view_pairs, final_extrinsics, K)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        box = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [SIZE, SIZE, SIZE])
        box.color = (0.1, 0.1, 0.1)
        o3d.visualization.draw_geometries(debug_clouds + [frame, box])

    detector = VGNImplicit(args.model, args.type, best=True,
                            force_detection=True, qual_th=args.qual_th,
                            out_th=0.5, visualize=False, resolution=80)
    grasps, scores, toc = detector(state, None)

    print(f"\n{len(grasps)} grasps found in {toc:.3f}s")
    for i, (g, s) in enumerate(zip(grasps, scores)):
        p_vol = g.pose.translation
        half_size = SIZE / 2.0
        p_task = p_vol - np.array([half_size, half_size, half_size])
        print(f"  [{i}] quality_score={s:.3f}")
        print(f"      Local TSDF Vol Pos = ({p_vol[0]:.3f}, {p_vol[1]:.3f}, {p_vol[2]:.3f})")
        print(f"      Actual Robot Task Pos = ({p_task[0]:.3f}, {p_task[1]:.3f}, {p_task[2]:.3f}) width={g.width:.3f}")

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
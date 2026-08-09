#!/usr/bin/env python3
"""
offline_test.py -- run entirely inside the IGD conda env. No ROS, no robot.

Loads one or more saved real depth frames + intrinsics + extrinsics, rebuilds
the state exactly like the sim, runs the IGD detector, and opens an Open3D
window with the scene cloud + the predicted grasps drawn as gripper
wireframes.

This is the single most important check before touching the robot: if the
grasps point sensibly INTO objects here, the whole perception path is correct.

Usage:
    conda activate igd_blackwell
    cd ~/Implicit-Grasp-Diffusion
    python offline_test.py \
        --depth /tmp/depth0.npy /tmp/depth1.npy \
        --fx 640.5 --fy 640.5 --cx 641.2 --cy 366.9 \
        --width 1280 --height 720 \
        --extrinsic /tmp/extrinsic0.npy /tmp/extrinsic1.npy \
        --model ./data/models/IGD_pile.pt --type igd \
        --icp --per-view-debug

--extrinsic is a 4x4 .npy mapping TASK-frame points -> CAMERA frame
(i.e. lookup_transform(camera_color_optical_frame, task)). See main().

--per-view-debug: before fusing, opens an Open3D window showing each view's
raw (unfused) point cloud in its own color, all in the same task frame. This
lets you visually check whether the extrinsics actually agree with each other
before you spend any time debugging the detector. Well-aligned views should
look like one coherent surface; misaligned views show doubled/ghosted edges.
If --icp is also given, the debug window is shown TWICE: once with the raw
extrinsics ("before"), once with the ICP-corrected extrinsics ("after"), so
you can see exactly what the correction did.

--icp: refine each view's extrinsic with ICP before fusing, using only the
TOP surface of whatever is in the scene (a fixed --icp-top-dist slab, default
1cm, measured down from the highest point in each view). This is deliberately
restricted to item tops rather than the whole cloud because tops are usually
the least-occluded,
least-noisy, most repeatable surface between viewing angles -- table/edge
points tend to pull a whole-cloud ICP fit in the wrong direction. One view
(--icp-reference-idx, default 0) is treated as the trusted anchor; every
other view's extrinsic is corrected to match it. This corrects small drift
in your given extrinsics; it is NOT a global registration step, so it will
not fix grossly wrong extrinsics (increase --icp-max-corr-dist a lot, or add
a RANSAC feature-matching prealignment, if views start out very far apart).
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


# ---------------------------------------------------------------------------
# ICP alignment, biased toward the TOP surface of whatever's in the scene.
# ---------------------------------------------------------------------------

def depth_to_camera_cloud(depth_m, K, depth_trunc=1.5):
    """Unproject a metres-scale depth image into a point cloud in the
    CAMERA frame (no extrinsic applied)."""
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(
        K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_m))
    return o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intr_o3d, depth_scale=1.0, depth_trunc=depth_trunc)


def top_slice(pc, dist=0.01, min_points=200):
    """Return only the points within `dist` metres of the highest z value
    in pc, i.e. a fixed-thickness slab off the top of whatever's in the
    scene, rather than the top-of-item surface. Falls back to keeping the
    min_points highest points if the slab is too thin, and returns pc
    unchanged if it's already too small to slice."""
    pts = np.asarray(pc.points)
    if len(pts) < min_points:
        return pc
    z = pts[:, 2]
    thresh = z.max() - dist
    mask = z >= thresh
    if mask.sum() < min_points:
        idx = np.argsort(z)[-min_points:]
        return pc.select_by_index(idx)
    return pc.select_by_index(np.where(mask)[0])


def icp_align_views(view_pairs, K, reference_idx=0, voxel_size=0.003,
                     max_corr_dist=0.015, top_dist=0.01,
                     point_to_plane=True, verbose=True):
    """
    Refine every view's extrinsic (except the reference view's) with ICP,
    using only the TOP-surface points of each view's cloud as the alignment
    target. Returns a new list of extrinsic matrices, same length/order as
    view_pairs.

    Why top-only: the top of an object is the surface most views actually
    agree on. Table edges, occlusion boundaries, and grazing-angle sides are
    noisy and inconsistent between viewpoints, and including them tends to
    drag a whole-cloud ICP fit off in some other direction. Restricting the
    correspondence search to a fixed-thickness slab off the top (top_dist
    metres below the highest point in each view) keeps the correction
    honest.
    """
    n = len(view_pairs)
    if n < 2:
        print("icp_align_views: only one view given, nothing to align.")
        return [ex for _, ex in view_pairs]

    # Build coarse task-frame clouds using the ORIGINAL extrinsics.
    task_clouds = []
    for depth_m, extrinsic_mat in view_pairs:
        cam_pc = depth_to_camera_cloud(depth_m, K)
        extr = Transform.from_matrix(extrinsic_mat)
        cam_to_task = extr.inverse().as_matrix()
        task_pc = o3d.geometry.PointCloud(cam_pc)
        task_pc.transform(cam_to_task)
        task_clouds.append(task_pc)

    ref_top = top_slice(task_clouds[reference_idx], dist=top_dist)
    ref_top = ref_top.voxel_down_sample(voxel_size)
    ref_top.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    refined = [None] * n
    refined[reference_idx] = view_pairs[reference_idx][1]

    estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane()
                  if point_to_plane else
                  o3d.pipelines.registration.TransformationEstimationPointToPoint())

    for i in range(n):
        if i == reference_idx:
            continue

        src_top = top_slice(task_clouds[i], dist=top_dist)
        src_top = src_top.voxel_down_sample(voxel_size)

        if len(src_top.points) < 30 or len(ref_top.points) < 30:
            print(f"  view [{i}]: too few top-surface points for ICP "
                  f"(src={len(src_top.points)}, ref={len(ref_top.points)}), "
                  f"keeping original extrinsic")
            refined[i] = view_pairs[i][1]
            continue

        src_top.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            src_top, ref_top, max_corr_dist, np.eye(4), estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))

        T_correction = result.transformation  # src(view i, old task frame) -> ref task frame

        if verbose:
            delta_t = T_correction[:3, 3]
            print(f"  view [{i}] ICP fitness={result.fitness:.3f} "
                  f"rmse={result.inlier_rmse:.5f} "
                  f"translation_correction=({delta_t[0]*1000:.1f}, "
                  f"{delta_t[1]*1000:.1f}, {delta_t[2]*1000:.1f}) mm")
            if result.fitness < 0.3:
                print(f"    WARNING: low fitness for view [{i}] -- ICP may "
                      f"not have converged to a sensible correction. Check "
                      f"--icp-max-corr-dist / --icp-top-dist, or inspect "
                      f"with --per-view-debug.")

        orig_extr_mat = view_pairs[i][1]
        T_correction_inv = np.linalg.inv(T_correction)
        # new_extr maps task_new(=reference task frame) -> camera_i
        new_extr_mat = orig_extr_mat @ T_correction_inv
        refined[i] = new_extr_mat

    return refined


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


def show_debug_window(view_pairs, K, label):
    print(f"--per-view-debug ({label}): building {len(view_pairs)} "
          f"unfused per-view clouds...")
    debug_clouds = per_view_clouds(view_pairs, K)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    box = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [SIZE, SIZE, SIZE])
    box.color = (0.1, 0.1, 0.1)
    print(f"Opening per-view debug window [{label}]. Well-aligned views "
          f"should look like one coherent surface; doubled/ghosted edges "
          f"mean an extrinsic (or timing) mismatch between views.")
    o3d.visualization.draw_geometries(debug_clouds + [frame, box])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", nargs="+", required=True)
    ap.add_argument("--extrinsic", nargs="+", required=True)
    ap.add_argument("--intrinsic", default=None,
                    help="path to <prefix>_intrinsic.npy written by view_capture.py "
                         "([width, height, fx, fy, cx, cy], read from camera_info). "
                         "Strongly preferred over passing --fx/--fy/--cx/--cy by hand: "
                         "the RealSense intrinsics change with the streaming profile, "
                         "and stale values silently rescale the reconstruction.")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
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

    # --- ICP alignment options ---
    ap.add_argument("--icp", action="store_true",
                    help="refine per-view extrinsics with ICP, aligned on "
                         "each view's top-surface points, before fusing")
    ap.add_argument("--icp-reference-idx", type=int, default=0,
                    help="index of the view treated as the trusted anchor; "
                         "all other views are corrected to match it")
    ap.add_argument("--icp-top-dist", type=float, default=0.01,
                    help="thickness (m) of the top slab used for ICP, "
                         "measured down from each view's highest point "
                         "(default 1cm) -- i.e. the tops of the items in "
                         "the scene")
    ap.add_argument("--icp-max-corr-dist", type=float, default=0.015,
                    help="ICP max correspondence distance in metres; only "
                         "fixes drift up to roughly this size")
    ap.add_argument("--icp-voxel", type=float, default=0.003,
                    help="voxel size (m) for downsampling before ICP")
    ap.add_argument("--icp-point-to-point", action="store_true",
                    help="use point-to-point ICP instead of the default "
                         "point-to-plane (point-to-plane is usually better "
                         "for the mostly-flat tops this targets)")

    args = ap.parse_args()

    if len(args.depth) != len(args.extrinsic):
        raise ValueError(f"Got {len(args.depth)} depth files but {len(args.extrinsic)} extrinsics — need one extrinsic per depth image, in matching order.")

    if args.intrinsic:
        w, h, fx, fy, cx, cy = np.load(args.intrinsic).astype(np.float64)
        K = dict(width=int(w), height=int(h), fx=fx, fy=fy, cx=cx, cy=cy)
        print(f"intrinsics from {args.intrinsic}: {int(w)}x{int(h)} "
              f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    else:
        missing = [n for n in ("fx", "fy", "cx", "cy", "width", "height")
                   if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"Need --intrinsic, or all of: {', '.join('--' + m for m in missing)}")
        K = dict(width=args.width, height=args.height,
                 fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    view_pairs = []
    for depth_path, extr_path in zip(args.depth, args.extrinsic):
        depth = np.load(depth_path).astype(np.float32)
        # A depth image whose size disagrees with the intrinsics reconstructs at
        # the wrong scale and off-centre, but still produces a plausible-looking
        # cloud -- so it has to be caught here rather than eyeballed later.
        if depth.shape != (K["height"], K["width"]):
            raise SystemExit(
                f"{depth_path} is {depth.shape[1]}x{depth.shape[0]} but the intrinsics "
                f"describe {K['width']}x{K['height']}. These must match: fx/fy/cx/cy are "
                f"in pixels and are only valid for the profile they were captured at.")
        if depth.max() > 100:
            depth = depth / args.depth_scale
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 1.2] = 0.0
        depth = remove_edge_flying_pixels(depth, grad_threshold=0.005)  # 0.02
        extr = np.load(extr_path).astype(np.float64)
        print(f"File: {extr_path} | Translation: {extr[:3, 3]}")
        view_pairs.append((depth, extr))

    if args.per_view_debug:
        label = "before ICP" if args.icp else "raw extrinsics"
        show_debug_window(view_pairs, K, label)

    if args.icp:
        if args.icp_reference_idx < 0 or args.icp_reference_idx >= len(view_pairs):
            raise ValueError(f"--icp-reference-idx {args.icp_reference_idx} out "
                              f"of range for {len(view_pairs)} views")
        print(f"Running ICP alignment (reference view = "
              f"{args.icp_reference_idx}, top_dist={args.icp_top_dist}m)...")
        refined_extrinsics = icp_align_views(
            view_pairs, K,
            reference_idx=args.icp_reference_idx,
            voxel_size=args.icp_voxel,
            max_corr_dist=args.icp_max_corr_dist,
            top_dist=args.icp_top_dist,
            point_to_plane=not args.icp_point_to_point)
        view_pairs = [(depth, refined_extrinsics[i])
                      for i, (depth, _) in enumerate(view_pairs)]

        if args.per_view_debug:
            show_debug_window(view_pairs, K, "after ICP")

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
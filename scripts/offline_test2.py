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


def centered_extrinsics(view_pairs, apply_centering=True):
    """
    extrinsic_mat @ T_tsdfCenter_to_tsdfOrigin for every view (raw, pre-fix).

    Set apply_centering=False to use extrinsics exactly as given, with no
    shift at all -- this matches what offline_test.py does. If your
    per-view-debug looks correct in offline_test.py but breaks in this
    script, A/B test --no-center-origin first before touching ICP: a
    SIZE/2 (15cm by default) origin mismatch will blow the scene apart
    just as badly as a broken ICP fix would, and looks similar at a
    glance (points scattered outside the workspace box, floor gone).
    """
    if not apply_centering:
        return [extr.copy() for _, extr in view_pairs]
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


def remove_dominant_plane(cloud, distance_threshold=0.004, ransac_n=3, num_iterations=1000,
                           expected_up_axis=(0.0, 0.0, 1.0), max_normal_angle_deg=25.0,
                           view_tag=""):
    """
    Segments out the largest planar surface in a point cloud (typically the
    table/floor the object sits on) via RANSAC, and returns only the
    remaining (non-planar) points.

    Why this matters for ICP: point-to-plane ICP on a scene dominated by one
    big flat surface is under-constrained *within* that surface -- sliding a
    view sideways along the table, or rotating it slightly about the
    surface normal, barely changes the point-to-plane residual, so the
    optimizer is free to pick any such in-plane offset. With thousands of
    floor points and only a few hundred object points, the optimizer will
    happily "solve" the alignment by sliding the floors past each other --
    which is exactly the parallel-stripe artifact you get from fusing the
    result. Removing the dominant plane before ICP forces the registration
    to rely on the object's actual (non-planar) geometry, which constrains
    all 6 DOF.

    SAFETY CHECK: RANSAC just finds the single largest flat surface -- it
    has no idea that surface is supposed to be the table. If your object
    has a big flat face of its own (a box, a book, a tray) and that face
    fills more of a given view than the visible table does, RANSAC will
    happily select the OBJECT's face as "the plane" and strip it out,
    leaving ICP with only noisy edge points to work with -- which produces
    exactly the kind of garbage, scattered-fragment alignment that looks
    like a much worse bug than it is. To guard against this, the detected
    plane's normal is checked against `expected_up_axis` (the table's
    known orientation in this frame); if it's off by more than
    `max_normal_angle_deg`, the "plane" is assumed to be a face of the
    object, not the table, and removal is skipped for this view.
    """
    if len(cloud.points) < max(ransac_n, 10):
        return cloud
    try:
        plane_model, inliers = cloud.segment_plane(
            distance_threshold=distance_threshold, ransac_n=ransac_n,
            num_iterations=num_iterations)
    except RuntimeError:
        return cloud

    a, b, c, _d = plane_model
    normal = np.array([a, b, c], dtype=float)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return cloud
    normal /= norm_len
    up = np.array(expected_up_axis, dtype=float)
    up /= np.linalg.norm(up)
    # abs(): a plane's normal sign from RANSAC is arbitrary (could point
    # either way), we only care about the axis it's aligned with.
    cos_angle = np.clip(abs(np.dot(normal, up)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    if angle_deg > max_normal_angle_deg:
        print(f"    {view_tag}[plane check] detected plane normal is {angle_deg:.1f} deg "
              f"from expected up-axis (> {max_normal_angle_deg} deg threshold), "
              f"{len(inliers)} inlier points -- this looks like a face of the OBJECT, "
              f"not the table. Skipping plane removal for this view.")
        return cloud

    print(f"    {view_tag}[plane check] detected plane normal is {angle_deg:.1f} deg "
          f"from expected up-axis, {len(inliers)} inlier points -- looks like the "
          f"table, removing it.")
    return cloud.select_by_index(inliers, invert=True)


def extract_top_slab(cloud, slab_thickness=0.02, min_points=50):
    """
    Keeps only the top slice of a point cloud: points within
    `slab_thickness` meters of the highest Z value present in the cloud.

    This is the "only match the top" idea. A tabletop object's top surface
    is (roughly) visible from every camera angle you use -- a top-down
    shot, and oblique left/right shots that both look down onto it -- so
    it is the one part of the object every view genuinely shares. The
    sides are a different story: your left view mostly sees the object's
    left face, your right view mostly sees its right face, and those two
    faces don't correspond to each other at all. Handing both full side
    clouds to ICP means it either has to find some sliver of legitimate
    overlap (small, noisy, easy to get a bad local optimum on) or it falls
    back onto whatever big shared-looking surface it *can* fit -- which
    is how you get the plane-sliding / streaking artifact even after
    plane removal, just from the *edges* of the leftover object cloud
    instead of the table.

    Restricting correspondences to a thin top band removes the
    non-corresponding side geometry entirely and gives ICP an
    unambiguous, genuinely-shared target in every view.

    NOTE: this assumes the convention used elsewhere in this file, where
    increasing Z in the centered task frame points *up* (see `floor` /
    `upper` bounds in build_state_aligned). If your extrinsics use a
    different convention, flip the sign or pass the axis explicitly.
    """
    pts = np.asarray(cloud.points)
    if len(pts) == 0:
        return cloud
    z_max = pts[:, 2].max()
    mask = pts[:, 2] >= (z_max - slab_thickness)
    if mask.sum() < min_points:
        return cloud
    return cloud.select_by_index(np.where(mask)[0])


def multiway_register(clouds, max_corr_coarse=0.03, max_corr_fine=0.008,
                       normal_radius=0.01, normal_max_nn=30,
                       remove_plane=True, plane_dist_threshold=0.004,
                       plane_max_normal_angle_deg=25.0,
                       top_slab_thickness=None, min_object_points=200,
                       min_pairwise_fitness=0.3):
    """
    Standard Open3D multiway registration: build a fully-connected pose
    graph from pairwise ICP, then globally optimize it.

    Returns a list of 4x4 "fix" transforms, one per view, each defined so
    that:
        p_fixed_i = fix[i] @ p_original_i
    where p_original_i are the points reconstructed from view i's own
    (uncorrected) extrinsic.

    Every pairwise ICP result is treated as *provisional*: only pairs whose
    fitness clears `min_pairwise_fitness` are added to the pose graph at
    all (as prunable, "uncertain" edges -- there is no notion of a trusted
    "odometry" edge here, since your views are independent viewpoints, not
    a smooth camera trajectory). A view with no edges clearing the bar is
    left at identity (no correction) rather than being pulled by a
    meaningless fit -- an unconstrained "correction" of tens of cm is worse
    than no correction at all.

    If `remove_plane` is True (default), the dominant table plane is
    stripped out of each view's cloud before it's used for ICP -- see
    `remove_dominant_plane` for why this matters.

    If `top_slab_thickness` is set (meters), the cloud used for ICP is
    further trimmed down to just the top slab of whatever's left after
    plane removal -- see `extract_top_slab`. Use this when your views are
    top + left + right (or similar), so side faces don't get matched
    against each other.

    The stripped/trimmed clouds are only used to *compute* the fix
    transforms; the final TSDF fusion still uses the full (floor + object)
    depth data.
    """
    n = len(clouds)
    if n < 2:
        return [np.eye(4) for _ in range(n)]

    reg_clouds = clouds
    if remove_plane:
        print("\nRemoving dominant (table) plane from each view before ICP:")
        trimmed_clouds = []
        for i, c in enumerate(reg_clouds):
            trimmed = remove_dominant_plane(c, distance_threshold=plane_dist_threshold,
                                             max_normal_angle_deg=plane_max_normal_angle_deg,
                                             view_tag=f"view[{i}] ")
            if len(trimmed.points) < min_object_points:
                print(f"  view[{i}]: only {len(trimmed.points)} points left after plane "
                      f"removal (< {min_object_points}) -- falling back to the full cloud "
                      f"for this view's ICP. Check the object is actually visible here, "
                      f"or lower --min-object-points / raise --plane-thresh.")
                trimmed = c
            else:
                print(f"  view[{i}]: {len(c.points)} -> {len(trimmed.points)} points "
                      f"after removing the dominant plane")
            trimmed_clouds.append(trimmed)
        reg_clouds = trimmed_clouds

    if top_slab_thickness is not None:
        print(f"\nRestricting ICP correspondences to the top {top_slab_thickness*1000:.0f}mm "
              f"of each view's cloud (top-only matching):")
        slab_clouds = []
        for i, c in enumerate(reg_clouds):
            slab = extract_top_slab(c, slab_thickness=top_slab_thickness)
            if len(slab.points) < min_object_points:
                print(f"  view[{i}]: only {len(slab.points)} points in top slab "
                      f"(< {min_object_points}) -- falling back to the pre-slab cloud "
                      f"for this view's ICP. Try a larger --top-slab-thickness, or check "
                      f"the object's top is actually captured in this view.")
                slab = c
            else:
                print(f"  view[{i}]: {len(c.points)} -> {len(slab.points)} points "
                      f"kept in top slab")
            slab_clouds.append(slab)
        reg_clouds = slab_clouds

    # Normals are required for point-to-plane ICP.
    for c in reg_clouds:
        c.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))

    # Every node starts at identity. Edges (added below) pull nodes into a
    # mutually consistent frame during optimization; a node touched by no
    # accepted edge simply stays at identity.
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for _ in range(n):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    print("\nPairwise ICP fit report (lower inlier_rmse = better; "
          "fitness closer to 1.0 = more overlap used):")
    has_edge = [False] * n
    n_edges_used = 0
    for source_id in range(n):
        for target_id in range(source_id + 1, n):
            transformation, information, fitness, rmse = pairwise_registration(
                reg_clouds[source_id], reg_clouds[target_id], max_corr_coarse, max_corr_fine)
            accepted = fitness >= min_pairwise_fitness
            status = "OK" if accepted else "REJECTED (fitness too low -- no real correspondence)"
            print(f"  view[{source_id}] <-> view[{target_id}]: "
                  f"fitness={fitness:.3f} inlier_rmse={rmse * 1000:.2f}mm "
                  f"proposed_correction={np.linalg.norm(transformation[:3, 3]) * 1000:.1f}mm  [{status}]")
            if not accepted:
                continue
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, transformation, information, uncertain=True))
            n_edges_used += 1
            has_edge[source_id] = True
            has_edge[target_id] = True

    if n_edges_used == 0:
        print("\nNo pairwise registration cleared the fitness threshold -- leaving all "
              "views uncorrected (identity fix). ICP has no reliable information to work "
              "with here; this points to a data problem (camera sync, depth scale, wrong "
              "extrinsic file, object out of frame) rather than something ICP tuning can fix.")
        return [np.eye(4) for _ in range(n)]

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    fixes = [pose_graph.nodes[i].pose for i in range(n)]

    print("\nFinal per-view correction magnitude (post pose-graph optimization):")
    for i, fix in enumerate(fixes):
        t = fix[:3, 3]
        tag = "  <-- no accepted edges touched this view; left uncorrected" if not has_edge[i] else ""
        print(f"  view[{i}]: translation_fix={np.linalg.norm(t) * 1000:.1f}mm "
              f"({t[0]*1000:+.1f}, {t[1]*1000:+.1f}, {t[2]*1000:+.1f})mm{tag}")
    return fixes


def diagnose_raw_alignment(view_pairs, K, plane_dist_threshold=0.004, min_object_points=50,
                            high_resolution=120, center_origin=True):
    """
    Standalone diagnostic (no ICP involved): reconstructs each view's cloud
    with its OWN uncorrected extrinsic, strips the dominant plane, and
    reports the centroid of what's left per view. This tells you how far
    apart the views actually place the object, independent of whether ICP
    itself can converge -- the number to compare against your ~2cm
    AprilTag estimate.
    """
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    base_extrinsics = centered_extrinsics(view_pairs, apply_centering=center_origin)
    clusters = []
    for i, (depth_m, _) in enumerate(view_pairs):
        raw = reconstruct_view_cloud(depth_m, intr, base_extrinsics[i], resolution=high_resolution)
        obj = remove_dominant_plane(raw, distance_threshold=plane_dist_threshold, view_tag=f"view[{i}] ")
        if len(obj.points) < min_object_points:
            print(f"  view[{i}]: only {len(obj.points)} object points -- too few to trust")
            clusters.append(None)
            continue
        centroid = np.asarray(obj.points).mean(axis=0)
        clusters.append((obj, centroid))
        print(f"  view[{i}]: {len(obj.points)} object points, centroid="
              f"({centroid[0]*1000:+.1f}, {centroid[1]*1000:+.1f}, {centroid[2]*1000:+.1f})mm")

    print("\nPairwise raw centroid distances (BEFORE any ICP correction):")
    for i in range(len(clusters)):
        if clusters[i] is None:
            continue
        for j in range(i + 1, len(clusters)):
            if clusters[j] is None:
                continue
            d = np.linalg.norm(clusters[i][1] - clusters[j][1]) * 1000
            flag = "  <-- much larger than a ~2cm calibration bias would explain" if d > 40 else ""
            print(f"  view[{i}] <-> view[{j}]: {d:.1f}mm{flag}")

    colored = []
    for i, c in enumerate(clusters):
        if c is None:
            continue
        obj, _ = c
        obj = o3d.geometry.PointCloud(obj)
        obj.paint_uniform_color(DEBUG_COLORS[i % len(DEBUG_COLORS)])
        colored.append(obj)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    box = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [SIZE, SIZE, SIZE])
    box.color = (0.1, 0.1, 0.1)
    o3d.visualization.draw_geometries(colored + [frame, box])


def build_state_aligned(view_pairs, K, resolution=80, high_resolution=120,
                         align=True, icp_coarse=0.03, icp_fine=0.008,
                         remove_plane=True, plane_dist_threshold=0.004,
                         plane_max_normal_angle_deg=25.0,
                         top_slab_thickness=None, min_object_points=200,
                         min_pairwise_fitness=0.3, center_origin=True):
    """
    Like your original build_state, but first ICP-aligns the per-view
    clouds and folds the resulting correction into each view's extrinsic
    before integrating the shared TSDF.

    center_origin controls whether extrinsics get the SIZE/2 centering
    shift (see centered_extrinsics). offline_test.py does NOT apply this
    shift -- if your per-view-debug looks right there but wrong here, set
    center_origin=False (--no-center-origin on the CLI) and compare again
    before touching anything ICP-related.
    """
    intr = CameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
    base_extrinsics = centered_extrinsics(view_pairs, apply_centering=center_origin)

    if align and len(view_pairs) > 1:
        # Reconstruct each view alone (at a reasonably high resolution) to
        # get clean clouds for ICP.
        raw_clouds = [
            reconstruct_view_cloud(depth_m, intr, base_extrinsics[i], resolution=high_resolution)
            for i, (depth_m, _) in enumerate(view_pairs)
        ]
        fixes = multiway_register(raw_clouds, max_corr_coarse=icp_coarse, max_corr_fine=icp_fine,
                                   remove_plane=remove_plane,
                                   plane_dist_threshold=plane_dist_threshold,
                                   plane_max_normal_angle_deg=plane_max_normal_angle_deg,
                                   top_slab_thickness=top_slab_thickness,
                                   min_object_points=min_object_points,
                                   min_pairwise_fitness=min_pairwise_fitness)
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
    ap.add_argument("--remove-plane", dest="remove_plane", action="store_true", default=True,
                     help="Strip the dominant table plane before computing ICP "
                          "corrections, so the object geometry drives alignment "
                          "instead of the plane (default: on)")
    ap.add_argument("--no-remove-plane", dest="remove_plane", action="store_false",
                     help="Disable plane removal (use full clouds, including the "
                          "table, for ICP -- likely to reproduce the streaking bug)")
    ap.add_argument("--plane-thresh", type=float, default=0.004,
                     help="RANSAC inlier distance (m) for detecting the table plane")
    ap.add_argument("--plane-max-angle", type=float, default=25.0,
                     help="Max angle (deg) between a detected plane's normal and the "
                          "expected up-axis (0,0,1) for it to be treated as the table. "
                          "Planes detected at a steeper angle are assumed to be a face "
                          "of the object itself and are left in place instead of being "
                          "stripped out. Raise this if your table isn't level in the "
                          "task frame; lower it if a flat object face is still getting "
                          "misidentified as the table.")
    ap.add_argument("--center-origin", dest="center_origin", action="store_true", default=True,
                     help="Apply the SIZE/2 centering shift to extrinsics before "
                          "reconstruction (default: on, matches this script's original "
                          "behavior)")
    ap.add_argument("--no-center-origin", dest="center_origin", action="store_false",
                     help="Use extrinsics exactly as given, with NO centering shift -- "
                          "this matches offline_test.py's convention. Try this first if "
                          "per-view-debug looks correct in offline_test.py but broken "
                          "here: a frame-origin mismatch can scatter the scene just as "
                          "badly as a broken ICP correction, before ICP is even involved.")
    ap.add_argument("--top-slab", dest="top_slab_thickness", type=float, default=None,
                     help="If set (meters), restrict ICP correspondences to the top "
                          "slab of each view's (plane-removed) cloud -- i.e. only "
                          "points within this distance of the highest point in that "
                          "view. Use this when your views are e.g. top + left + right "
                          "and the side faces don't overlap with each other. A "
                          "starting value around 0.015-0.02 (15-20mm) is reasonable "
                          "for small tabletop objects; try --diagnose-only first to "
                          "see typical object height and pick a slab that stays well "
                          "within it.")
    ap.add_argument("--min-object-points", type=int, default=200,
                     help="If fewer than this many non-plane (or non-slab) points "
                          "remain in a view, fall back to the previous, less-trimmed "
                          "cloud for that view's ICP")
    ap.add_argument("--min-fitness", type=float, default=0.3,
                     help="Minimum ICP fitness for a pairwise result to be trusted and "
                          "added to the pose graph at all; pairs below this are dropped "
                          "rather than corrupting the correction for other views")
    ap.add_argument("--diagnose-only", action="store_true",
                     help="Skip ICP/TSDF/grasp detection entirely. Just reconstruct each "
                          "view's own (uncorrected) object cluster, print pairwise raw "
                          "centroid distances, and show them color-coded. Use this first "
                          "if corrections look too large -- it tells you how far apart the "
                          "views actually disagree, independent of whether ICP converges.")
    args = ap.parse_args()

    if len(args.depth) != len(args.extrinsic):
        raise ValueError("Mismatched pairs between input image sequences and extrinsic configurations.")

    K = dict(width=args.width, height=args.height,
             fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

    view_pairs = load_views(args.depth, args.extrinsic, depth_scale=args.depth_scale)

    if args.diagnose_only:
        print(f"Diagnosing raw (uncorrected) alignment across {len(view_pairs)} views...")
        diagnose_raw_alignment(view_pairs, K, plane_dist_threshold=args.plane_thresh,
                                min_object_points=max(50, args.min_object_points // 4),
                                center_origin=args.center_origin)
        return

    print(f"Aligning {len(view_pairs)} views before merging "
          f"({'ICP correction ON' if args.align else 'ICP correction OFF'})...")
    state, pc, final_extrinsics = build_state_aligned(
        view_pairs, K, resolution=80, high_resolution=120,
        align=args.align, icp_coarse=args.icp_coarse, icp_fine=args.icp_fine,
        remove_plane=args.remove_plane, plane_dist_threshold=args.plane_thresh,
        plane_max_normal_angle_deg=args.plane_max_angle,
        top_slab_thickness=args.top_slab_thickness,
        min_object_points=args.min_object_points, min_pairwise_fitness=args.min_fitness,
        center_origin=args.center_origin)

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
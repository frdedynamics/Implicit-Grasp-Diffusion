# #!/usr/bin/env python3
# """
# Recomputes fr3_link8 -> camera_link so it can be published WITHOUT conflicting
# with the RealSense driver's own camera_link -> ... -> camera_color_optical_frame
# static chain (camera_color_optical_frame can only have ONE parent in TF).

# Your easy_handeye2 calibration gave you fr3_link8 -> camera_color_optical_frame
# directly. This script composes that with the driver's own (already published)
# camera_link -> camera_color_optical_frame transform to back out
# fr3_link8 -> camera_link, which is what you should actually publish.
# """
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.time import Time
# from rclpy.duration import Duration
# from rclpy.executors import SingleThreadedExecutor
# import tf2_ros


# def quat_to_mat(x, y, z, w):
#     n = x*x + y*y + z*z + w*w
#     s = 0.0 if n < 1e-12 else 2.0 / n
#     xx, yy, zz = x*x*s, y*y*s, z*z*s
#     xy, xz, yz = x*y*s, x*z*s, y*z*s
#     wx, wy, wz = w*x*s, w*y*s, w*z*s
#     return np.array([[1-(yy+zz), xy-wz, xz+wy],
#                       [xy+wz, 1-(xx+zz), yz-wx],
#                       [xz-wy, yz+wx, 1-(xx+yy)]])


# def mat_to_quat(R):
#     tr = R[0, 0] + R[1, 1] + R[2, 2]
#     if tr > 0:
#         S = np.sqrt(tr + 1.0) * 2
#         w = 0.25 * S
#         x = (R[2, 1] - R[1, 2]) / S
#         y = (R[0, 2] - R[2, 0]) / S
#         z = (R[1, 0] - R[0, 1]) / S
#     elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
#         S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
#         w = (R[2, 1] - R[1, 2]) / S
#         x = 0.25 * S
#         y = (R[0, 1] + R[1, 0]) / S
#         z = (R[0, 2] + R[2, 0]) / S
#     elif R[1, 1] > R[2, 2]:
#         S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
#         w = (R[0, 2] - R[2, 0]) / S
#         x = (R[0, 1] + R[1, 0]) / S
#         y = 0.25 * S
#         z = (R[1, 2] + R[2, 1]) / S
#     else:
#         S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
#         w = (R[1, 0] - R[0, 1]) / S
#         x = (R[0, 2] + R[2, 0]) / S
#         y = (R[1, 2] + R[2, 1]) / S
#         z = 0.25 * S
#     return x, y, z, w


# def make_T(x, y, z, qx, qy, qz, qw):
#     T = np.eye(4)
#     T[:3, :3] = quat_to_mat(qx, qy, qz, qw)
#     T[:3, 3] = [x, y, z]
#     return T


# def main():
    
#     T_link8_cam = make_T(
#         0.006640680368234056, -0.06059790303750654, 0.06074720832487268,
#         -0.0035160834148036204, 0.0016469755130771313,
#         0.3606233711862283, 0.9327034409625385
#     )

#     rclpy.init()
#     node = Node("recompute_camera_link_tf")
    
#     # Setup executor to cleanly pump TF updates
#     executor = SingleThreadedExecutor()
#     executor.add_node(node)
    
#     buf = tf2_ros.Buffer()
#     # Explicitly pass the node to the listener
#     tf2_ros.TransformListener(buf, node)

#     tf = None
#     print("Waiting for RealSense TF frames...")
#     for _ in range(200):
#         # Spin the executor briefly to let subscriptions process incoming frames
#         executor.spin_once(timeout_sec=0.05)
#         try:
#             # Look up camera_color_optical_frame relative to camera_link
#             tf = buf.lookup_transform("camera_link", "camera_color_optical_frame",
#                                        Time())
#             break
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
#             continue

#     if tf is None:
#         print("Failed to look up camera_link -> camera_color_optical_frame.")
#         print("Is the RealSense driver running?")
#         node.destroy_node()
#         rclpy.shutdown()
#         return

#     t = tf.transform.translation
#     q = tf.transform.rotation
#     T_camlink_cam = make_T(t.x, t.y, t.z, q.x, q.y, q.z, q.w)

#     # Transform inverted correctly: 
#     # T_(link8 -> camera_link) = T_(link8 -> cam) * T_(cam -> camera_link)
#     T_link8_camlink = T_link8_cam @ np.linalg.inv(T_camlink_cam)

#     x, y, z = T_link8_camlink[:3, 3]
#     qx, qy, qz, qw = mat_to_quat(T_link8_camlink[:3, :3])

#     print("\nPublish THIS instead (fr3_link8 -> camera_link):\n")
#     print("ros2 run tf2_ros static_transform_publisher \\")
#     print(f"  --x {x:.6f} --y {y:.6f} --z {z:.6f} \\")
#     print(f"  --qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f} \\")
#     print("  --frame-id fr3_link8 --child-frame-id camera_link")

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Recomputes fr3_link8 -> camera_link based on an optical-frame calibration.

Your easy_handeye2 calibration should be run using:
  tracking_base_frame:=camera_color_optical_frame

This script grabs the active, hardware-defined baseline between camera_link 
and camera_color_optical_frame from the running RealSense driver, combines 
it with your calibration matrix, and outputs the exact static transform 
you need to publish for your physical mounting point.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import SingleThreadedExecutor
import tf2_ros


def quat_to_mat(x, y, z, w):
    n = x*x + y*y + z*z + w*w
    s = 0.0 if n < 1e-12 else 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([[1-(yy+zz), xy-wz, xz+wy],
                      [xy+wz, 1-(xx+zz), yz-wx],
                      [xz-wy, yz+wx, 1-(xx+yy)]])


def mat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return x, y, z, w


def make_T(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(qx, qy, qz, qw)
    T[:3, 3] = [x, y, z]
    return T


def main():
    # =========================================================================
    # REPLACE THE VALUES BELOW WITH YOUR FRESH CALIBRATION DATA
    # Generated using: tracking_base_frame:=camera_color_optical_frame
    # =========================================================================
    T_link8_cam = make_T(
        0.005323182497440505, -0.06236424331366505, 0.061376882383208334,  # x, y, z
        -0.003753682570275934, 0.0005473652684778729, 0.352849080143383, 0.9356725585910876 # qx, qy, qz, qw
    )
    # =========================================================================

    rclpy.init()
    node = Node("recompute_camera_link_tf")
    
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    tf = None
    print("Waiting for RealSense TF frames...")
    
    # Attempt to read the factory transform baseline from the driver
    for _ in range(200):
        executor.spin_once(timeout_sec=0.05)
        try:
            # We explicitly ask for camera_color_optical_frame relative to camera_link
            tf = buf.lookup_transform("camera_link", "camera_color_optical_frame", Time())
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

    if tf is None:
        print("Error: Failed to look up camera_link -> camera_color_optical_frame.")
        print("Please verify that your RealSense driver node is currently active and running.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Extract translation and rotation from the driver's TF
    t = tf.transform.translation
    q = tf.transform.rotation
    T_camlink_camcolor = make_T(t.x, t.y, t.z, q.x, q.y, q.z, q.w)

    # Compute: T_(link8 -> camera_link) = T_(link8 -> camera_color_optical) * T_(camera_color_optical -> camera_link)
    # T_(camera_color_optical -> camera_link) is the inverse of T_(camera_link -> camera_color_optical)
    T_link8_camlink = T_link8_cam @ np.linalg.inv(T_camlink_camcolor)

    # Extract vector and quaternions for the command output
    x, y, z = T_link8_camlink[:3, 3]
    qx, qy, qz, qw = mat_to_quat(T_link8_camlink[:3, :3])

    print("\nCalculated successfully! Publish this command to bridge your TF tree:")
    print("\nros2 run tf2_ros static_transform_publisher \\")
    print(f"  --x {x:.6f} --y {y:.6f} --z {z:.6f} \\")
    print(f"  --qx {qx:.6f} --qy {qy:.6f} --qz {qz:.6f} --qw {qw:.6f} \\")
    print("  --frame-id fr3_link8 --child-frame-id camera_link")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

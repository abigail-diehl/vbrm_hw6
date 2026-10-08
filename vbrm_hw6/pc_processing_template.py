"""Student starter: ROS PointCloud2 -> PCL -> grasp pose in base_link.

Implement process_cloud() and estimate_grasp(). The supplied node publishes
preview clouds and candidate poses only; it never commands robot motion.
Clouds are pcl_python_merlab.PointCloud objects. Coordinates are in meters.
The course-local binding runs real PCL algorithms with the system Python.
NumPy and sensor_msgs_py are used at the ROS message / TF boundaries.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from pcl_python_merlab import PointCloud
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


def cloud_from_msg(msg):
    xyz = point_cloud2.read_points_numpy(
        msg, field_names=('x', 'y', 'z'), skip_nans=False,
    ).reshape(-1, 3)
    return PointCloud(xyz[np.isfinite(xyz).all(axis=1)].astype(np.float32))


def transform_cloud(cloud, transform):
    """Apply the TF pose using PCL; return a new PointCloud."""
    q = transform.transform.rotation
    t = transform.transform.translation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError('TF rotation must be a finite, nonzero quaternion')
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (t.x, t.y, t.z)
    return cloud.transform(matrix)


class PointCloudGrasping(Node):
    def __init__(self):
        super().__init__('pc_processing_template')
        self.declare_parameter('point_cloud_topic', '/camera/points')
        self.declare_parameter('point_cloud_topic_2', '/camera2/points')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('processing_period_sec', 0.5)
        self.declare_parameter('voxel_size', 0.005)

        self.declare_parameter('table_planeZ', 0.01)
        self.declare_parameter('table_planeX', 0.1)

        self.table_planeZ = self.get_parameter('table_planeZ').value
        self.table_planeX = self.get_parameter('table_planeX').value


        self.target_frame = self.get_parameter('target_frame').value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        period = float(self.get_parameter('processing_period_sec').value)
        if (not np.isfinite(self.voxel_size) or self.voxel_size <= 0.0
                or not np.isfinite(period) or period <= 0.0):
            raise ValueError('voxel_size and processing_period_sec must be finite and positive')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_cloud = None
        self.latest_cloud2 = None
        
        self.subscription = self.create_subscription(
            PointCloud2, self.get_parameter('point_cloud_topic').value,
            self.cloud_callback, qos_profile_sensor_data,
        )
        self.cloud_publisher = self.create_publisher(
            PointCloud2, '~/processed_cloud', 1,
        )


        #cam 2 PC sub/pub: 
        self.subscription = self.create_subscription(
            PointCloud2, self.get_parameter('point_cloud_topic_2').value,
            self.cloud_callback2, qos_profile_sensor_data,
        )

        self.grasp_publisher = self.create_publisher(PoseStamped, '~/grasp_pose', 1)
        # Cache only the latest frame; process at a manageable rate.
        self.timer = self.create_timer(period, self.process_latest_cloud)

        self.get_logger().info(f"PC node started!")


    def cloud_callback(self, msg):
        self.latest_cloud = msg


    def cloud_callback2(self, msg):
        self.latest_cloud2 = msg

    def process_latest_cloud(self):
        msg = self.latest_cloud
        msg2 = self.latest_cloud2
        if msg is None or msg2 is None:
            # self.get_logger().info(f"waiitng for PC")
            return
        
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id,Time.from_msg(msg.header.stamp),
            )

            transform2 = self.tf_buffer.lookup_transform(self.target_frame, msg2.header.frame_id,Time.from_msg(msg2.header.stamp)
            )
        except TransformException as exc:
            return
        
        self.latest_cloud = None
        self.latest_cloud2 = None
        
        cloud = cloud_from_msg(msg)
        cloud2 = cloud_from_msg(msg2)
        
        if len(cloud) == 0 or len(cloud2) == 0:
            self.get_logger().info(f"cloud length = 0")
            return

        # self.get_logger().info(f"length of cloud 1: {len(cloud)}")
        # self.get_logger().info(f"length of cloud 2: {len(cloud2)}")

        
        cloud = transform_cloud(cloud, transform)
        cloud2 = transform_cloud(cloud2, transform2)

        #merging the PCs: 
        merged = cloud.concatenate(cloud2)
        
        processed = self.process_cloud(merged)

        if len(processed) == 0:
            self.get_logger().info(f"processed cloud length = 0")
            return

        #Part 1.1 ends here with publishing the combined PC. 

        #Part 1.2: Filtering the WS: 
        filtered = processed.pass_through(axis="z", minimum = self.table_planeZ, maximum = 5.0)
        filtered = filtered.pass_through(axis="x", minimum = self.table_planeX, maximum = 20.0)

        #Part 1.3: Segment the Table Frame: 
        inliers, coefficients = filtered.segment_plane(distance_threshold = 5, max_iterations = 50)
        if not inliers: 
            self.get_logger().info(f"fitting failed")
            return 1

        segmented = cloud.extract(inliers)

        self.get_logger().info(f"segmented PC!")
        
        header = Header(stamp=msg.header.stamp, frame_id=self.target_frame)

        self.cloud_publisher.publish(point_cloud2.create_cloud_xyz32(header, segmented.xyz))

        pose = self.estimate_grasp(segmented)

        if pose is not None :
            self.grasp_publisher.publish(PoseStamped(header=header, pose=pose))

    def process_cloud(self, cloud):
        cloud = cloud.voxel_grid(self.voxel_size)


        return cloud

    def estimate_grasp(self, cloud) -> Pose | None:

        #TODO: estimate the grasp from the PC

        return None


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGrasping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

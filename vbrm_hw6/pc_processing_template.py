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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


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

        self.centroid_publisher = self.create_publisher(Marker, '~/centroid', 1)
        self.normals_publisher = self.create_publisher(Marker, '~/normals', 1)
        self.grasp_points_publisher = self.create_publisher(Marker, '~/grasp_points', 1)


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
        inliers, coefficients = self.find_horizontal_plane(filtered)

        if inliers is None: 
            self.get_logger().info(f"Could not find horizontal plane")
            return 

        segmented = filtered.extract(inliers)

        #Part 1.4: Segment the Objects: 

        segmentedObjects = filtered.extract(inliers, negative = True)


        #Part 1.5: Cluster the objs
        cylinderObject = self.find_cylinder_object(segmentedObjects)

        #Part 2.1: Calculate the Obj Centroid
        centroid = self.find_cylinder_center(cylinderObject)

        #Part 2.2: Estimate surface normals

        surface_norms = self.get_surface_normals(cylinderObject, centroid)

        grasp_pair = self.find_grasp_pair(cylinderObject, surface_norms, centroid)


        header = Header(stamp = msg.header.stamp, frame_id=self.target_frame)
        self.cloud_publisher.publish(point_cloud2.create_cloud_xyz32(header, cylinderObject.xyz))

        pose = self.estimate_grasp(segmented)

        if pose is not None :
            self.grasp_publisher.publish(PoseStamped(header=header, pose=pose))

    def process_cloud(self, cloud):
        cloud = cloud.voxel_grid(self.voxel_size)


        return cloud


    def estimate_grasp(self, cloud) -> Pose | None:

        #TODO: estimate the grasp from the PC: 

    
        return None

    def find_grasp_pair(self, cloud, normals, centroid):
        points = cloud.xyz

        #threshold???
        normal_dot_threshold = -0.9
        best_pair = None
        best_distance = float('inf')

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1 = points[i]
                p2 = points[j]

                n1 = normals[i]
                n2 = normals[j]

                normal1 = np.linalg.norm(n1)
                normal2 = np.linalg.norm(n2)

                if normal1 == 0 or normal2 == 0:
                    continue

                dot = np.dot(n1 / normal1, n2 / normal2)
                if dot > normal_dot_threshold:
                    continue

                distance = (np.linalg.norm(p1 - centroid) + np.linalg.norm(p2 - centroid))

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i,j)

        if best_pair is None: 
            self.get_logger().info(f"can't find a grasp pair :( ")
            return None

        i,j = best_pair

        p1 = points[i]
        p2 = points[j]

        #publishing to rViz, generated: 
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'grasp_points'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD


        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03


        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        for p in (p1, p2):
            point = Point()
            point.x = float(p[0])
            point.y = float(p[1])
            point.z = float(p[2])
            marker.points.append(point)

        self.grasp_points_publisher.publish(marker)

        self.get_logger().info(f"found grasp pair!")

        return p1, p2

    def get_surface_normals(self, cloud, viewpoint):
        viewpoint = [0.0, 0.0, 1.0]
        normals = cloud.estimate_normals(radius = 0.3)
        # self.get_logger().info(f"normal array shape: {normals.shape}")

        valid = np.isfinite(normals).all(axis=1)
        valid &= np.linalg.norm(normals, axis = 1) > 1e-8

        usableCloud = cloud.extract(np.flatnonzero(valid).tolist())
        usableNormals = normals[valid]

        if len(usableCloud) == 0: 
            raise RuntimeError("No usable normals :(")

        withCurvature = cloud.normals_with_curvature(
            radius = 0.03, viewpoint = viewpoint)

        #publishing: generated this part
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'surface_norms'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD


        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.03

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        line_length = 0.03
        for point, normal in zip(cloud.xyz, normals):
            start = Point()
            start.x = float(point[0])
            start.y = float(point[1])
            start.z = float(point[2])

            end = Point()
            end.x = float(point[0] + line_length * normal[0])
            end.y = float(point[1] + line_length * normal[1])
            end.z = float(point[2] + line_length * normal[2])

            marker.points.append(start)
            marker.points.append(end)


        self.normals_publisher.publish(marker)

        return withCurvature



    def find_cylinder_center(self, cloud):
            centroid = np.mean(cloud.xyz, axis=0)
    
            # Publishing: generated this print
    
            marker = Marker()
    
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
    
            marker.ns = 'centroid'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
    
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
    
            marker.pose.orientation.w = 1.0
    
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03
    
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
    
            self.centroid_publisher.publish(marker)
            # self.get_logger().info(f"found centroid: {centroid}")
    
            return centroid
    
    def find_cylinder_object(self, cloud):
        normals = cloud.estimate_normals(radius = 0.02)
        valid = np.isfinite(normals).all(axis =1 )
        valid &= np.linalg.norm(normals, axis=1) > 1e-8

        cloud = cloud.extract(np.flatnonzero(valid).tolist())
        normals = normals[valid]

        plane_indices, plane_coefficients = cloud.segment_plane(distance_threshold=0.03, max_iterations=100)

        if not plane_indices:
            raise RuntimeError(f"no plane found :(")

        plane = cloud.extract(plane_indices)

        keep = np.ones(len(cloud), dtype=bool)
        keep[plane_indices] = False
        remaining = cloud.extract(np.flatnonzero(keep).tolist())
        remaining_normals = normals[keep]

        if len(remaining) < 3: 
            raise RuntimeError(f"Not enough for a cylinder!")

        indices, coefficients = remaining.segment_cylinder(
            normals = remaining_normals, 
            distance_threshold = 0.05,
            min_radius = 0.0,
            max_radius = 0.1,
            max_iterations = 10000,
            normal_distance_weight = 0.1
        )

        if not indices: 
            raise RuntimeError(f"No cyliner found :( ")

        cylinder = remaining.extract(indices)

        if cylinder:
            return cylinder

        return None

    def find_horizontal_plane(self, cloud):

        curr_cloud = cloud

        for i in range(10):

            if len(curr_cloud) < 3:
                break

            # Find the largest plane in the current cloud
            inliers, coefficients = curr_cloud.segment_plane(
                distance_threshold=0.005,
                max_iterations=50
            )

            # self.get_logger().info(
            #     f"Plane {i}: {len(inliers)} inliers"
            # )

            if not inliers:
                break

            # Plane equation:
            # ax + by + cz + d = 0
            a, b, c, d = coefficients

            normal = np.array(
                [a, b, c],
                dtype=np.float64
            )

            norm = np.linalg.norm(normal)

            if norm == 0:
                break

            # Normalize normal
            normal /= norm

            # Horizontal plane should have normal parallel to Z
            z_alignment = abs(normal[2])

            # self.get_logger().info(
            #     f"normal = {normal}, "
            #     f"z_alignment = {z_alignment:.3f}"
            # )

            # Found horizontal plane
            if z_alignment >= np.cos(np.deg2rad(15.0)):

                # self.get_logger().info(
                #     "found it!"
                # )

                return inliers, coefficients

            # This plane is not horizontal.
            # Remove it and search again.
            all_indices = np.arange(len(curr_cloud))

            inlier_set = set(inliers)

            remaining_indices = [
                idx for idx in all_indices
                if idx not in inlier_set
            ]

            curr_cloud = curr_cloud.extract(remaining_indices)

        return None, None


                
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

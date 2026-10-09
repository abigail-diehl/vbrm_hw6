import rclpy
import cv2
import numpy as np
from pathlib import Path
from common_interfaces_merlab.srv import SendJointTrajectoryPoint, SendPose, SendTwist
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point, PoseStamped
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image
import time
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

class GraspCylinder(Node):
    def __init__(self):
        super().__init__("grasp_cylinder")

        self.cartesian_client = self.create_client(SendPose, '/cartesian_ref')
        self.joint_client = self.create_client(SendJointTrajectoryPoint, 'move_traj_single_point')
        self.capture_client = self.create_client(Trigger, '/capture_image')

        self.declare_parameter("velocity_service", "/set_ee_velocity")
        velocity_service = self.get_parameter("velocity_service").value
        self.ee_velocity_client = self.create_client(SendTwist, velocity_service)
        self.velocity_service_name = velocity_service


        self.grasp_pose = None
        self.approach_vector = None

        self.grasp_pose_subscriber = self.create_subscription(PoseStamped, '/grasp_pose', self.grasp_pose_callback, 1)
        self.appraoach_msg_subscriber = self.create_subscription(Vector3Stamped, '/approach_msg', self.approach_msg_callback, 1)

        self.pregrasp_started = False

        self.gripper_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller_gripper/commands', 10)


    def grasp_pose_callback(self, msg):
        self.grasp_pose = msg
        # self.get_logger().info(f"recieved grasp pose!")
        self.pregrasp_pose()

    def approach_msg_callback(self, msg):
        self.approach_vector = msg
        # self.get_logger().info(f"recieved approach msg!")
        self.pregrasp_pose()

    def pregrasp_response_callback(self, future):
        try:
            response = future.result()
            if response is None: 
                self.get_logger().error((f"pregrasp service returned no response"))
                return

            self.get_logger().info(f"cartesian service:  {response}")
            
            if response.success:
                self.get_logger().info(f"pregrasp pose reached")

            else:
                self.get_logger().error(f"pregrasp pose not reached")

        except Exception as error:
            self.get_logger().error(f"pregrasp pose failed: {error}")

    def pregrasp_pose(self):

        if self.grasp_pose is None or self.approach_vector is None:
            return False

        if not self.cartesian_client.service_is_ready():
            self.get_logger().error(f"/cartesian_ref isn't ready")
            self.pregrasp_started = False

        if self.pregrasp_started:
            return

        self.pregrasp_started = True
        self.get_logger().info(f"starting pregrasp pose")


        grasp = self.grasp_pose.pose    


        vector = self.approach_vector.vector

        direction = np.array([vector.x, vector.y, vector.z], dtype = float)

        #testing if it is pointing the wrong way:
        # direction = -direction

        # direction = np.array([0.0, 0.0, -1.0])

        magnitute = np.linalg.norm(direction)

        if magnitute < 1e-6:
            self.get_logger().info(f"no mag on the approach")
            return None

        direction = direction / magnitute

        offset = 0.3
        pregrasp = PoseStamped()
        pregrasp.pose.position.x = grasp.position.x + offset * direction[0]
        pregrasp.pose.position.y = grasp.position.y + offset * direction[1]
        pregrasp.pose.position.z = grasp.position.z + offset * direction[2]

        pregrasp.pose.orientation = grasp.orientation

        # #from hw5: 
        # qx, qy, qz, qw = R.from_euler("xyz", [3.00, 0.15, 0.2]).as_quat()

        #  gripper orientation for a top-down grasp?
       
        # pregrasp.pose.orientation.x = float(qx)
        # pregrasp.pose.orientation.y = float(qy)
        # pregrasp.pose.orientation.z = float(qz)
        # pregrasp.pose.orientation.w = float(qw)

        # pregrasp.pose.orientation.x = 0.0
        # pregrasp.pose.orientation.y = 0.707
        # pregrasp.pose.orientation.z = 0.0
        # pregrasp.pose.orientation.w = 0.707


        self.get_logger().info(
            f"Pre-grasp target position: "
            f"x={pregrasp.pose.position.x:.3f}, "
            f"y={pregrasp.pose.position.y:.3f}, "
            f"z={pregrasp.pose.position.z:.3f}"
        )

        self.get_logger().info(
            f"Pre-grasp target orientation: "
            f"x={pregrasp.pose.orientation.x:.3f}, "
            f"y={pregrasp.pose.orientation.y:.3f}, "
            f"z={pregrasp.pose.orientation.z:.3f}, "
            f"w={pregrasp.pose.orientation.w:.3f}"
        )

        request = SendPose.Request()
        request.pose = pregrasp.pose

        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.pregrasp_response_callback)

        return True


def main(args=None):
    rclpy.init(args=args)
    node = GraspCylinder()
    try:
        node.get_logger().info(f"waiting for /grasp_posee and /approach_msg")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
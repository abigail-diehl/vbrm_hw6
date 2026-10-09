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
import copy
import time

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

        self.sequence_started = False
        self.pregrasp_started = False
        self.grasp_started = False

        self.pregrasp_target = None
        self.grasp_target = None

        self.offset = 0.3

        self.list_timer = None

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


        except Exception as error:
            self.get_logger().error(f"pregrasp pose failed: {error}")

        if response is None or not response.success:
            self.get_logger().info(f'pregrasp pose reached!')

        
        self.pregrasp_started = False
        self.grasp_started = True
        self.grasp_cylinder()

    def pregrasp_pose(self):

        if self.grasp_pose is None or self.approach_vector is None:
            return False

        if not self.cartesian_client.service_is_ready():
            self.get_logger().error(f"/cartesian_ref isn't ready")
            self.pregrasp_started = False
            return False

        if self.sequence_started:
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

        grasp.position.z += 0.2

        grasp = copy.deepcopy(self.grasp_pose.pose)
        self.grasp_target = copy.deepcopy(grasp)

        pregrasp = PoseStamped()
        pregrasp.pose.position.x = grasp.position.x + self.offset * direction[0]
        pregrasp.pose.position.y = grasp.position.y + self.offset * direction[1]
        pregrasp.pose.position.z = grasp.position.z + self.offset * direction[2]

        pregrasp.pose.orientation = grasp.orientation

        self.pregrasp_target = copy.deepcopy(pregrasp.pose)

        self.grasp_target = copy.deepcopy(grasp)

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


        # self.get_logger().info(
        #     f"Pre-grasp target position: "
        #     f"x={pregrasp.pose.position.x:.3f}, "
        #     f"y={pregrasp.pose.position.y:.3f}, "
        #     f"z={pregrasp.pose.position.z:.3f}"
        # )

        # self.get_logger().info(
        #     f"Pre-grasp target orientation: "
        #     f"x={pregrasp.pose.orientation.x:.3f}, "
        #     f"y={pregrasp.pose.orientation.y:.3f}, "
        #     f"z={pregrasp.pose.orientation.z:.3f}, "
        #     f"w={pregrasp.pose.orientation.w:.3f}"
        # )

        self.sequence_started = True
        self.pregrasp_started = True

        request = SendPose.Request()
        request.pose = pregrasp.pose

        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.pregrasp_response_callback)

        return True


    def move_gripper(self, position):
        #close gripper to preset gripper position
        close = Float64MultiArray()
        close.data = [position] 
        self.gripper_pub.publish(close)
        time.sleep(1.5)

        self.get_logger().info("closing gripper")

    def lift_response_callback(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f'lift failed!')
            self.grasp_started = False
            return 

        if response is None or not response.success:
            self.get_logger().error(f'failed post-grasp pose')
            self.grasp_started = False
            return

        self.get_logger().info(f'post-grasp completed!')
        self.grasp_started = False

    def grasp_response_callback(self, future):
        try: 
            response = future.result()
        except:
            self.get_logger().error(f'grasp failed :(')
            self.grasp_started = False
            return 

        if response is None or not response.success:
            self.get_logger().error('grasp position not reached, wont cloes grip')
            self.grasp_started = False
            return

        self.get_logger().info(f'grasp position reached!')

        self.move_gripper(0.7) #number from hw5 / move_to_start! 
        request = SendPose.Request()
        request.pose = self.pregrasp_target

        self.get_logger().info(f'moving to post-grasp pose!')
        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.lift_response_callback)

    def grasp_cylinder(self):

        if not self.grasp_started:
            return False


        if not self.cartesian_client.service_is_ready():
            self.get_logger().error(f'/cartesian _ref isnt responding')
            self.grasp_started = False
            return False

        #translate in -z by offset amt 

        request = SendPose.Request()
        request.pose = self.grasp_target

        self.get_logger().info(f'moving to grasp pose!')

        future = self.cartesian_client.call_async(request)
        future.add_done_callback(self.grasp_response_callback)



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
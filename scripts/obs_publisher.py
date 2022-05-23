#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import pickle


class StateObsPublisher():
    def __init__(self, link_name, marker_link_name):
        self.link_name = link_name
        self.marker_link_name = marker_link_name

        self._initial_state_found = False
        self._initial_finger_found = False
        self._initial_obj_found = False

        self.eef_pos = None
        self.eef_quaternion = None
        self.finger_joints = np.zeros(2)
        self.target_pos = None
        self.obj_pos = None
        self.goal = np.array([0.4, -0.2, 0.425])
        
        self.listener = tf.TransformListener()
        rospy.Subscriber("franka_state_controller/franka_states",
                         FrankaState, self.franka_state_callback)
        rospy.Subscriber("franka_gripper/joint_states", 
                         JointState, self.franka_gripper_state_callback)
        rospy.Subscriber("cartesian_ik_controller/equilibrium_pose", 
                         PoseStamped, self.target_pose_callback)
        self.rgb_image = None
        self.bridge = CvBridge()
        rospy.Subscriber("camera/color/image_raw", Image, self.rgb_callback)
        self.step_count = 0
        while not (self._initial_state_found and self._initial_finger_found):
            rospy.sleep(1)
        
        # Box center in marker frame
        self.m_T_center = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -0.025],
                [0, 0, 0, 1]
            ])
        self.obs_pub = rospy.Publisher("rl_observation", Float32MultiArray, queue_size=10)
        rospy.Timer(rospy.Duration(0.05), 
                    lambda msg: self.marker_tf_callback(msg))

    def run(self):
        # run pose publisher
        rospy.Timer(rospy.Duration(0.1),
                    lambda msg: self.publisherCallback(msg))
    
    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.rgb_image = np.asarray(cv_image)
    
    def publisherCallback(self, msg):
        if self.eef_pos is not None and self.obj_pos is not None:
            if self.target_pos is None:
                self.target_pos = self.eef_pos.copy()
            observation = np.concatenate(
                [self.obj_pos, self.eef_pos, self.eef_quaternion, self.finger_joints, 
                 self.target_pos, self.goal])
            # Huang Tao's input config
            # observation = np.concatenate(
            #     [obs_obj_pose[0], obs_goal, obs_eef_pose[0], obs_eef_pose[1], [obs_finger_width],
            #     obs_obj_pose[0], tf.transformations.euler_from_quaternion(obs_obj_pose[1])]
            # )
            observation = Float32MultiArray(data=observation)
            print("publish", observation)
            saved_data = {"image": self.rgb_image, "q": self.robot_q, "eef_pos": self.eef_pos, 
                      "finger_width": self.finger_joints, "box": self.obj_pos, "observation": observation}
            with open("/home/yunfei/Documents/real_data/%d.pkl" % self.step_count, "wb") as f:
                pickle.dump(saved_data, f)
            self.step_count += 1
            self.obs_pub.publish(observation)
        else:
            print("observation not ready")
            print("obj_pos", self.obj_pos, "eef_pos", self.eef_pos)

    def franka_state_callback(self, msg):
        O_T_EE = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))
        eef_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
        self.eef_quaternion = eef_quaternion / np.linalg.norm(eef_quaternion)
        self.eef_pos = np.array(tf.transformations.translation_from_matrix(O_T_EE)) + np.array([0, 0, 0.4])
        self.robot_q = np.array(msg.q)
        if not self._initial_state_found:
            self._initial_state_found = True
    
    def franka_gripper_state_callback(self, msg):
        self.finger_joints[0] = msg.position[0]
        self.finger_joints[1] = msg.position[1]
        if not self._initial_finger_found:
            self._initial_finger_found = True

    def target_pose_callback(self, msg):
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) + np.array([0, 0, 0.4])

    def marker_tf_callback(self, msg):
        try:
            tvec, rvec = self.listener.lookupTransform(self.link_name, self.marker_link_name, rospy.Time())
        except:
            print(self.listener.getFrameStrings())
            print("try to look up", self.link_name, self.marker_link_name)
            return
        # todo: timestamp, avoid out of date data
        # rvec = np.array([0, 0, 0, 1])
        
        O_T_marker = tf.transformations.quaternion_matrix(rvec)
        O_T_marker[:3, 3] = np.array(tvec)
        # O_T_marker[1, 3] -= 0.05
        # O_T_marker[0, 3] += 0.01
        O_T_center = np.matmul(O_T_marker, self.m_T_center)
        self.obj_pos = O_T_center[:3, 3] + np.array([0., 0., 0.4])
        # print("obj pos", self.obj_pos)
        # obs_obj_pose = (O_T_center[:3, 3], rvec)


if __name__ == "__main__":
    rospy.init_node("obs_publisher_node")
    link_name = rospy.get_param("~link_name")
    marker_link_name = rospy.get_param("~marker_link_name")
    obs_publisher = StateObsPublisher(link_name, marker_link_name)
    obs_publisher.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")

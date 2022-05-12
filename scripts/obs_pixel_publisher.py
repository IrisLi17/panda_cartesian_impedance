#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped


class ObsPublisher():
    def __init__(self):
        self._initial_state_found = False
        self._initial_finger_found = False
        self._initial_image_found = False
        self.eef_pos = np.zeros(3)
        self.eef_quaternion = np.zeros(4)
        self.finger_joints = np.zeros(2)
        self.target_pos = None
        self.box_goal = np.array([0.3, 0.0, 0.425])
        self._image_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self._image_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        self.normalized_rgb = np.zeros(3 * 224 * 224)
        self.bridge = CvBridge()
        rospy.Subscriber("franka_state_controller/franka_states",
                         FrankaState, self.franka_state_callback)
        rospy.Subscriber("franka_gripper/joint_states", 
                         JointState, self.franka_gripper_state_callback)
        rospy.Subscriber("camera/color/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("cartesian_ik_controller/equilibrium_pose", PoseStamped, self.target_pose_callback)
        self.obs_pub = rospy.Publisher(
            "rl_observation", Float32MultiArray, queue_size=10)
        while not (self._initial_state_found and self._initial_finger_found and self._initial_image_found):
            rospy.sleep(1)
    
    def run(self):
        def publisherCallback(msg):
            if self.target_pos is None:
                self.target_pos = self.eef_pos.copy()
            observation = np.concatenate(
                [self.normalized_rgb, self.eef_pos, self.eef_quaternion, self.finger_joints, self.target_pos, self.box_goal])
            print(observation[-15:])
            observation = Float32MultiArray(data=observation)
            self.obs_pub.publish(observation)
    
        # run pose publisher
        self.timer = rospy.Timer(rospy.Duration(0.1),
                    lambda msg: publisherCallback(msg))
    
    def stop(self):
        self.timer.shutdown()
    
    def franka_state_callback(self, msg):
        O_T_EE = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))
        F_T_EE = np.transpose(np.reshape(msg.F_T_EE, (4, 4)))
        # print("F_T_EE", F_T_EE) # check if this is static
        # [[ 0.70709997,  0.70709997,  0.        ,  0.        ],
        # [-0.70709997,  0.70709997,  0.        ,  0.        ],
        # [ 0.        ,  0.        ,  1.        ,  0.1034    ],
        # [ 0.        ,  0.        ,  0.        ,  1.        ]]
        eef_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
        self.eef_quaternion = eef_quaternion / np.linalg.norm(eef_quaternion)
        self.eef_pos = np.array(tf.transformations.translation_from_matrix(O_T_EE)) + np.array([0, 0, 0.4])
        if not self._initial_state_found:
            self._initial_state_found = True
    
    def franka_gripper_state_callback(self, msg):
        self.finger_joints[0] = msg.position[0]
        self.finger_joints[1] = msg.position[1]
        if not self._initial_finger_found:
            self._initial_finger_found = True
    
    def target_pose_callback(self, msg):
        self.target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) + np.array([0, 0, 0.4])

    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # Resize, crop, normalize
        cv_image = cv2.resize(cv_image, (149, 84), interpolation=cv2.INTER_LINEAR)
        cv_image = cv_image[0: 84, 32: 32 + 84]
        cv2.imshow("cropped", cv_image)
        cv2.waitKey(1)
        np_image = np.asarray(cv_image)
        # assert np_image.shape == (224, 224, 3), np_image.shape
        # print(np_image.shape, np_image[105:110, 105: 110])
        self.normalized_rgb = np.transpose((np_image / 255.0 - self._image_mean) / self._image_std, (2, 0, 1)).reshape(-1)
        if not self._initial_image_found:
            self._initial_image_found = True


if __name__ == "__main__":
    rospy.init_node("obs_publisher_node")
    obs_publisher = ObsPublisher()
    obs_publisher.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    cv2.destroyAllWindows()
    obs_publisher.stop()

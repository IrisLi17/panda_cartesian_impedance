#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState

listener = None
obs_obj_pose = None
obs_eef_pose = None
obs_finger_width = None
obs_goal = np.array([0.3, 0.0, 0.025])
initial_pose_found = False
pose_pub = None
step_count = 0
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]


def franka_state_callback(msg):
    O_T_EE = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))
    eef_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
    eef_quaternion = eef_quaternion / np.linalg.norm(eef_quaternion)
    eef_pos = tf.transformations.translation_from_matrix(O_T_EE)
    global obs_eef_pose
    obs_eef_pose = (eef_pos, eef_quaternion)
    if not initial_pose_found:
        global initial_pose_found
        initial_pose_found = True


def franka_gripper_state_callback(msg):
    global obs_finger_width
    obs_finger_width = msg.position[0] + msg.position[1]


def marker_tf_callback(msg, ref_link_name, marker_link_name):
    try:
        tvec, rvec = listener.lookupTransform(ref_link_name, marker_link_name, rospy.Time())
    except:
        return
    global obs_obj_pose
    # todo: timestamp, avoid out of date data
    tvec[2] = 0.025
    # rvec = np.array([0, 0, 0, 1])
    m_T_center = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.025],
            [0, 0, 0, 1]
        ])
    O_T_marker = tf.transformations.quaternion_matrix(rvec)
    O_T_marker[:3, 3] = np.array(tvec)
    O_T_marker[1, 3] -= 0.00
    O_T_marker[0, 3] -= 0.01
    O_T_center = np.matmul(O_T_marker, m_T_center)
    obs_obj_pose = (O_T_center[:3, 3], rvec)


def publisherCallback(msg):
    if obs_eef_pose is not None and obs_finger_width is not None and obs_obj_pose is not None:
        # observation = np.concatenate(
        #     [obs_eef_pose[0], obs_eef_pose[1], [obs_finger_width], 
        #     obs_obj_pose[0], tf.transformations.euler_from_quaternion(obs_obj_pose[1]),
        #     obs_obj_pose[0], obs_goal])
        observation = np.concatenate(
            [obs_obj_pose[0], obs_goal, obs_eef_pose[0], obs_eef_pose[1], [obs_finger_width],
            obs_obj_pose[0], tf.transformations.euler_from_quaternion(obs_obj_pose[1])]
        )
        observation = Float32MultiArray(data=observation)
        obs_pub.publish(observation)
    else:
        print("observation not ready")


if __name__ == "__main__":
    rospy.init_node("obs_publisher_node")
    state_sub = rospy.Subscriber("franka_state_controller/franka_states",
                                 FrankaState, franka_state_callback)
    rospy.Subscriber("franka_gripper/joint_states", 
                     JointState, franka_gripper_state_callback)
    listener = tf.TransformListener()
    link_name = rospy.get_param("~link_name")
    marker_link_name = rospy.get_param("~marker_link_name")

    # Get initial pose for the interactive marker
    while not initial_pose_found:
        rospy.sleep(1)
    
    rospy.Timer(rospy.Duration(0.05), 
                lambda msg: marker_tf_callback(msg, link_name, marker_link_name))
    
    obs_pub = rospy.Publisher(
        "rl_observation", Float32MultiArray, queue_size=10)
    
    # run pose publisher
    rospy.Timer(rospy.Duration(0.05),
                lambda msg: publisherCallback(msg))

    rospy.spin()

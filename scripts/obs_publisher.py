#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState

listener = None
obs_obj_pose = None
obs_eef_pose = None
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


def marker_tf_callback(msg, ref_link_name, marker_link_name):
    try:
        tvec, rvec = listener.lookupTransform(ref_link_name, marker_link_name, rospy.Time())
    except:
        return
    global obs_obj_pose
    obs_obj_pose = (tvec, rvec)


def publisherCallback(msg):
    observation = np.concatenate(
        [obs_eef_pose[0], tf.transformations.euler_from_quaternion(obs_eef_pose[1]),
         obs_obj_pose[0], tf.transformations.euler_from_quaternion(obs_obj_pose[1])])
    obs_pub.publish(observation)


if __name__ == "__main__":
    rospy.init_node("obs_publisher_node")
    state_sub = rospy.Subscriber("franka_state_controller/franka_states",
                                 FrankaState, franka_state_callback)
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

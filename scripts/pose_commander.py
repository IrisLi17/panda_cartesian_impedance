#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import math

from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState

marker_pose = PoseStamped()
initial_pose_found = False
pose_pub = None
step_count = 0
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]


def publisherCallback(msg, link_name):
    if step_count < 500:
        marker_pose.header.frame_id = link_name
        marker_pose.header.stamp = rospy.Time(0)
        marker_pose.pose.position.x = 0.5 + 0.2 * math.sin(step_count / 50.0 * math.pi)
        marker_pose.pose.position.y = 0.3 * math.sin(step_count / 100.0 * math.pi)
        marker_pose.pose.orientation.x = 1.0
        marker_pose.pose.orientation.y = 0.0
        marker_pose.pose.orientation.z = 0.0
        marker_pose.pose.orientation.w = 0.0
        pose_pub.publish(marker_pose)
        global step_count
        step_count += 1


def franka_state_callback(msg):
    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(msg.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
    marker_pose.pose.orientation.x = initial_quaternion[0]
    marker_pose.pose.orientation.y = initial_quaternion[1]
    marker_pose.pose.orientation.z = initial_quaternion[2]
    marker_pose.pose.orientation.w = initial_quaternion[3]
    marker_pose.pose.position.x = msg.O_T_EE[12]
    marker_pose.pose.position.y = msg.O_T_EE[13]
    marker_pose.pose.position.z = msg.O_T_EE[14]
    global initial_pose_found
    initial_pose_found = True


if __name__ == "__main__":
    rospy.init_node("pose_commander_node")
    state_sub = rospy.Subscriber("franka_state_controller/franka_states",
                                 FrankaState, franka_state_callback)
    listener = tf.TransformListener()
    link_name = rospy.get_param("~link_name")

    # Get initial pose for the interactive marker
    while not initial_pose_found:
        rospy.sleep(1)
    state_sub.unregister()

    pose_pub = rospy.Publisher(
        "equilibrium_pose", PoseStamped, queue_size=10)
    
    # run pose publisher
    rospy.Timer(rospy.Duration(0.05),
                lambda msg: publisherCallback(msg, link_name))

    rospy.spin()

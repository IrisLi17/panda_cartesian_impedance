#!/usr/bin/env python

import rospy
import tf.transformations
import actionlib
import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from panda_cartesian_impedance.msg import DualPoseStamped
from franka_msgs.msg import FrankaState
import franka_gripper.msg

dual_eef_pose = DualPoseStamped()
left_initial_pose_found = False
right_initial_pose_found = False
pose_pub = None
left_gripper_homing_client = None
left_gripper_move_client = None
left_gripper_grasp_client = None
right_gripper_homing_client = None
right_gripper_move_client = None
right_gripper_grasp_client = None
step_count = 0
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]


def action_callback(msg, left_link_name, right_link_name):
    action = msg.data
    # If should grasp or release
    # move_gripper_to
    # gripper_grasp
    # Parse dual eef pose
    pose_pub.publish(dual_eef_pose)

# This is an example
def publisherCallback(msg, left_link_name, right_link_name):
    if step_count < 500:
        dual_eef_pose.left_pose.header.frame_id = left_link_name
        dual_eef_pose.left_pose.header.stamp = rospy.Time(0)
        dual_eef_pose.left_pose.pose.position.x = 0.5 + 0.2 * math.sin(step_count / 50.0 * math.pi)
        dual_eef_pose.left_pose.pose.position.y = 0.3 * math.sin(step_count / 100.0 * math.pi)
        dual_eef_pose.left_pose.pose.orientation.x = -math.sqrt(2) / 2.0
        dual_eef_pose.left_pose.pose.orientation.y = math.sqrt(2) / 2.0
        dual_eef_pose.left_pose.pose.orientation.z = 0.0
        dual_eef_pose.left_pose.pose.orientation.w = 0.0
        
        dual_eef_pose.right_pose.header.frame_id = right_link_name
        dual_eef_pose.right_pose.header.stamp = rospy.Time(0)
        dual_eef_pose.right_pose.pose.position.x = 0.5
        dual_eef_pose.right_pose.pose.position.y = 0.3 * math.sin(step_count / 100.0 * math.pi)
        dual_eef_pose.right_pose.pose.position.z = 0.4 + 0.1 * math.sin(step_count / 50.0 * math.pi)
        dual_eef_pose.right_pose.pose.orientation.x = 1.0
        dual_eef_pose.right_pose.pose.orientation.y = 0.0
        dual_eef_pose.right_pose.pose.orientation.z = 0.0
        dual_eef_pose.right_pose.pose.orientation.w = 0.0
        
        rospy.loginfo(step_count)
        pose_pub.publish(dual_eef_pose)

        if step_count % 50 == 0:
            move_gripper_to(None, 1, 0.08 * step_count / 500.0)
            move_gripper_to(None, 2, 0.08 - 0.08 * step_count / 500.0)
        
        global step_count
        step_count += 1


def franka_state_callback(msg, id):
    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(msg.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
    if id == 1:
        O_T_EE_pose = dual_eef_pose.left_pose
    elif id == 2:
        O_T_EE_pose = dual_eef_pose.right_pose
    else:
        raise RuntimeError
    O_T_EE_pose.pose.orientation.x = initial_quaternion[0]
    O_T_EE_pose.pose.orientation.y = initial_quaternion[1]
    O_T_EE_pose.pose.orientation.z = initial_quaternion[2]
    O_T_EE_pose.pose.orientation.w = initial_quaternion[3]
    O_T_EE_pose.pose.position.x = msg.O_T_EE[12]
    O_T_EE_pose.pose.position.y = msg.O_T_EE[13]
    O_T_EE_pose.pose.position.z = msg.O_T_EE[14]
    if id == 1:
        global left_initial_pose_found
        left_initial_pose_found = True
    elif id == 2:
        global right_initial_pose_found
        right_initial_pose_found = True


def move_gripper_to(msg, id, width, speed=0.1, block=True):
    goal = franka_gripper.msg.MoveGoal(width=width, speed=speed)
    if id == 1:
        client = left_gripper_move_client
    elif id == 2:
        client = right_gripper_move_client
    else:
        raise RuntimeError
    if block:
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(3)) # TODO: timeout
    else:
        client.send_goal(goal)


def gripper_homing(msg, id, block=True):
    goal = franka_gripper.msg.HomingGoal()
    if id == 1:
        client = left_gripper_homing_client
    elif id == 2:
        client = right_gripper_homing_client
    else:
        raise RuntimeError
    if block:
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(3))
    else:
        client.send_goal(goal)

def gripper_grasp(msg, id, width, speed=0.05, force=10, inner_epsilon=0.0, outer_epsilon=0.01, block=True):
    epsilon = franka_gripper.msg.GraspEpsilon(inner=inner_epsilon, outer=outer_epsilon)
    goal = franka_gripper.msg.GraspGoal(
        width=width, speed=speed, epsilon=epsilon, force=force)
    if id == 1:
        client = left_gripper_grasp_client
    elif id == 2:
        client = right_gripper_grasp_client
    if block:
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(3))
    else:
        client.send_goal(goal)


if __name__ == "__main__":
    rospy.init_node("rl_commander_node")
    state_left_sub = rospy.Subscriber("panda_1_state_controller/franka_states",
                                      FrankaState, franka_state_callback, 1)
    state_right_sub = rospy.Subscriber("panda_2_state_controller/franka_states",
                                       FrankaState, franka_state_callback, 2)
    listener = tf.TransformListener()
    left_link_name = rospy.get_param("~left_link_name")
    right_link_name = rospy.get_param("~right_link_name")

    left_gripper_homing_client = actionlib.SimpleActionClient(
        '/panda_1/franka_gripper/homing', franka_gripper.msg.HomingAction)
    left_gripper_move_client = actionlib.SimpleActionClient(
        '/panda_1/franka_gripper/move', franka_gripper.msg.MoveAction)
    left_gripper_grasp_client = actionlib.SimpleActionClient(
        '/panda_1/franka_gripper/grasp', franka_gripper.msg.GraspAction)
    right_gripper_homing_client = actionlib.SimpleActionClient(
        '/panda_2/franka_gripper/homing', franka_gripper.msg.HomingAction)
    right_gripper_move_client = actionlib.SimpleActionClient(
        '/panda_2/franka_gripper/move', franka_gripper.msg.MoveAction)
    right_gripper_grasp_client =actionlib.SimpleActionClient(
        '/panda_2/franka_gripper/grasp', franka_gripper.msg.GraspAction)

    rospy.loginfo("Waiting for gripper homing server")
    left_gripper_homing_client.wait_for_server()
    right_gripper_homing_client.wait_for_server()

    rospy.loginfo("Waiting for gripper move server")
    left_gripper_move_client.wait_for_server()
    right_gripper_move_client.wait_for_server()

    rospy.loginfo("Waiting for gripper grasp server")
    left_gripper_grasp_client.wait_for_server()
    right_gripper_grasp_client.wait_for_server()

    # Get initial pose for the interactive marker
    while (not left_initial_pose_found) or (not right_initial_pose_found):
        rospy.sleep(1)
    rospy.loginfo(dual_eef_pose)
    state_left_sub.unregister()
    state_right_sub.unregister()

    action_sub = rospy.Subscriber(
        "rl_action", Float32MultiArray, 
        action_callback, left_link_name, right_link_name
    )

    pose_pub = rospy.Publisher(
        "equilibrium_pose", DualPoseStamped, queue_size=10)
    
    rospy.spin()

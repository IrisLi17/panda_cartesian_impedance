#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import math
import actionlib
from collections import deque

from geometry_msgs.msg import PoseStamped, Pose
from franka_msgs.msg import FrankaState
import franka_gripper
import franka_gripper.msg
from sensor_msgs.msg import JointState


class ExpertController(object):
    def __init__(self):
        self.desired_pose = PoseStamped()
        self._initial_pose_found = False
        self.eef_pos = np.zeros(3)
        self.box_pos_obs = deque(maxlen=5)
        self.box_pos = np.array([0.4, 0.0, 0.025])
        self.m_T_center = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.025],
            [0, 0, 0, 1]
        ])
        self.finger_width = 0.08
        state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states",
            FrankaState, self.franka_state_callback)
        rospy.Subscriber("franka_gripper/joint_states", 
                    JointState, self.franka_gripper_state_callback)
        self.tf_listener = tf.TransformListener()
        self.link_name = rospy.get_param("~link_name")
        self.marker_link = rospy.get_param("~marker_link")
        # Get initial pose for the interactive marker
        while not self._initial_pose_found:
            rospy.sleep(1)
        # Initialize gripper clients
        self.gripper_homing_client = actionlib.SimpleActionClient(
          'franka_gripper/homing', franka_gripper.msg.HomingAction)
        self.gripper_move_client = actionlib.SimpleActionClient(
            'franka_gripper/move', franka_gripper.msg.MoveAction)
        self.gripper_grasp_client = actionlib.SimpleActionClient(
            'franka_gripper/grasp', franka_gripper.msg.GraspAction)
        
        print("Waiting for gripper homing server")
        self.gripper_homing_client.wait_for_server()

        print("Waiting for gripper move server")
        self.gripper_move_client.wait_for_server()

        print("Waiting for gripper grasp server")
        self.gripper_grasp_client.wait_for_server()

        self.pose_pub = rospy.Publisher(
            "equilibrium_pose", PoseStamped, queue_size=10)
        self.phase = -1 # for control
        self.gripper_grasp_lock = 0
        self.gripper_open_lock = 0
        # run pose publisher
        rospy.Timer(rospy.Duration(0.1),
                    lambda msg: self.control_callback(msg))
    
    def franka_state_callback(self, msg):
        initial_quaternion = \
            tf.transformations.quaternion_from_matrix(
                np.transpose(np.reshape(msg.O_T_EE,
                                        (4, 4))))
        initial_quaternion = initial_quaternion / np.linalg.norm(initial_quaternion)
        if not self._initial_pose_found:
            self.desired_pose.pose.orientation.x = initial_quaternion[0]
            self.desired_pose.pose.orientation.y = initial_quaternion[1]
            self.desired_pose.pose.orientation.z = initial_quaternion[2]
            self.desired_pose.pose.orientation.w = initial_quaternion[3]
            self.desired_pose.pose.position.x = msg.O_T_EE[12]
            self.desired_pose.pose.position.y = msg.O_T_EE[13]
            self.desired_pose.pose.position.z = msg.O_T_EE[14]
            self._initial_pose_found = True
        else:
            self.eef_pos[0] = msg.O_T_EE[12]
            self.eef_pos[1] = msg.O_T_EE[13]
            self.eef_pos[2] = msg.O_T_EE[14]
    
    def franka_gripper_state_callback(self, msg):
        self.finger_width = msg.position[0] + msg.position[1]
    
    def control_callback(self, msg):
        try:
            tvec, rvec = self.tf_listener.lookupTransform(self.link_name, self.marker_link, rospy.Time())
            O_T_marker = tf.transformations.quaternion_matrix(rvec)
            O_T_marker[:3, 3] = np.array(tvec)
            # O_T_marker[1, 3] -= 0.00
            # O_T_marker[0, 3] -= 0.00
            O_T_center = np.matmul(O_T_marker, self.m_T_center)
            self.box_pos_obs.append(O_T_center[:3, 3])
            self.box_pos = np.mean(np.stack(self.box_pos_obs, axis=0), axis=0)
            print(O_T_marker[:3, 3], self.box_pos)
            if self.phase == -1:
                self.phase = 0
        except:
            print("Box pos not received")
        if self.phase == 0:
            dpos = np.clip(self.box_pos + np.array([0, 0, 0.1]) - self.eef_pos, -0.05 ,0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            self.desired_pose.pose.orientation.x = 1.0
            self.desired_pose.pose.orientation.y = 0.0
            self.desired_pose.pose.orientation.z = 0.0
            self.desired_pose.pose.orientation.w = 0.0
            print("In phase", self.phase, "hand pos", self.eef_pos, "error", np.linalg.norm(dpos))
            if np.linalg.norm(dpos) < 1e-2:
                self.phase = 1
        elif self.phase == 1:
            dpos = np.clip(self.box_pos - self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            print("In phase", self.phase, "hand pos", self.eef_pos, "error", np.linalg.norm(dpos))
            if np.linalg.norm(dpos) < 1e-2:
                self.phase = 2
        elif self.phase == 2:
            if not self.gripper_grasp_lock:
                epsilon = franka_gripper.msg.GraspEpsilon(inner=0.01, outer=0.01)
                goal = franka_gripper.msg.GraspGoal(
                    width=0.05, speed=0.1, epsilon=epsilon, force=1)

                self.gripper_grasp_client.send_goal(goal)
                self.gripper_grasp_lock = 1
                self.gripper_grasp_client.wait_for_result(rospy.Duration(0.1))
            print("In phase", self.phase, self.finger_width)
            if self.gripper_grasp_lock and self.finger_width < 0.055:
                self.phase = 3
                self.gripper_grasp_lock = 0
                self.up_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] + 0.1]
        elif self.phase == 3:
            # epsilon = franka_gripper.msg.GraspEpsilon(inner=0.01, outer=0.01)
            # goal = franka_gripper.msg.GraspGoal(
            #     width=0.05, speed=0.1, epsilon=epsilon, force=1)
            # self.gripper_grasp_client.send_goal(goal)
            self.desired_pose.pose.position.x = self.up_pos[0]
            self.desired_pose.pose.position.y = self.up_pos[1]
            self.desired_pose.pose.position.z = self.up_pos[2]
            print("In phase", self.phase, "hand pos", self.eef_pos)
            if abs(self.eef_pos[2] - self.up_pos[2]) < 3e-2:
                self.gripper_grasp_client.cancel_all_goals()
                self.phase = 4
        elif self.phase == 4:
            if not self.gripper_open_lock:
                goal = franka_gripper.msg.MoveGoal(width=0.08, speed=0.1)
                self.gripper_move_client.send_goal(goal)
                print("In phase", self.phase, self.finger_width)
                self.gripper_open_lock = 1

            if self.gripper_open_lock and self.finger_width > 0.078:
                self.phase = 0
                self.gripper_move_client.cancel_all_goals()
                self.gripper_open_lock = 0
        self.pose_pub.publish(self.desired_pose)


if __name__ == "__main__":
    rospy.init_node("pose_commander_node")
    controller = ExpertController()
    rospy.spin()
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


class PoseCommander():
    def __init__(self):
        self.desired_pose = PoseStamped()
        self.eef_pos = np.zeros(3)
        self._initial_pose_found = False
        state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states",
            FrankaState, self.franka_state_callback)
        while not self._initial_pose_found:
            rospy.sleep(1)
        self.pose_pub = rospy.Publisher(
            "equilibrium_pose", PoseStamped, queue_size=10)
        self.test_points = []
        self.achieved_points = []
        
    def start(self):
        # run pose publisher
        self.timer = rospy.Timer(rospy.Duration(0.5),
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
    
    def control_callback(self, msg):
        if len(self.achieved_points) > 100:
            self.test_points = np.stack(self.test_points[:-1], axis=0)
            self.achieved_points = np.stack(self.achieved_points, axis=0)
            errors = np.linalg.norm(self.test_points - self.achieved_points, axis=-1)
            print("Error mean", np.mean(errors), "Error min", np.min(errors), "Error max", np.max(errors))
            self.timer.shutdown()
        else:
            if len(self.test_points) > 0:
                self.achieved_points.append(self.eef_pos.copy())
            test_point = self.eef_pos.copy()
            test_point[0] += np.random.uniform(-0.03, 0.03)
            test_point[1] += np.random.uniform(-0.03, 0.03)
            test_point[2] += np.random.uniform(-0.03, 0.03)
            test_point[0] = np.clip(test_point[0], 0.25, 0.6)
            test_point[1] = np.clip(test_point[1], -0.4, 0.4)
            test_point[2] = np.clip(test_point[2], 0.05, 0.7)
            self.test_points.append(test_point.copy())
            self.desired_pose.pose.position.x = test_point[0]
            self.desired_pose.pose.position.y = test_point[1]
            self.desired_pose.pose.position.z = test_point[2]
            self.desired_pose.pose.orientation.x = 1.0
            self.desired_pose.pose.orientation.y = 0.0
            self.desired_pose.pose.orientation.z = 0.0
            self.desired_pose.pose.orientation.w = 0.0
            self.pose_pub.publish(self.desired_pose)
            
if __name__ == "__main__":
    np.random.seed(42)
    rospy.init_node("test_accuracy_node")
    pose_commander = PoseCommander()
    pose_commander.start()
    rospy.spin()
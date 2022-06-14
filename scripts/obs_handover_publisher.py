#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from std_msgs.msg import Float32MultiArray
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from aruco_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from collections import deque


class StateObsPublisher():
    def __init__(self, link_name, markers_topic):
        self.link_name = link_name
        self.markers_topic = markers_topic

        self._initial_state_found = False
        self._initial_finger_found = False
        self._initial_obj_found = False

        self.eef_pos = None
        self.eef_quaternion = None
        self.eef_vel = np.array([0., 0., 0.])
        self.finger_joints = np.zeros(2)
        self.target_pos = None
        self.obj1_pos = None
        self.obj1_pos_obs = deque(maxlen=10)
        self.obj2_pos = None
        self.obj2_pos_obs = deque(maxlen=10)
        self.goal1 = np.array([-0.45, 0.0, 0.42])
        self.goal2 = np.array([-0.3 ,-0.2, 0.42])
        self.offset = None
        self.goal_mean = np.array([0.0, 0.0, 0.52])
        self.goal_std = np.array([0.55, 0.15, 0.1])

        self.marker1_id = [8, 9]
        self.marker2_id = [18, 16]
        
        self.marker_right_id = []
        self.marker_left_id = []
        self.tf_listener = tf.TransformListener()
        rospy.Subscriber("franka_state_controller/franka_states",
                         FrankaState, self.franka_state_callback)
        rospy.Subscriber("franka_gripper/joint_states", 
                         JointState, self.franka_gripper_state_callback)
        self.step_count = 0
        while not (self._initial_state_found and self._initial_finger_found):
            rospy.sleep(1)
        
        # Box center in marker frame
        self.m_T_center = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -0.02],
                [0, 0, 0, 1]
            ])
        self.mr_T_center = np.array([
            [1, 0, 0, -0.044],
            [0, 1, 0, 0],
            [0, 0, 1, -0.02],
            [0, 0, 0, 1]
        ])
        self.ml_T_center = np.array([
            [1, 0, 0, 0.045],
            [0, 1, 0 ,0 ],
            [0, 0, 1, -0.02],
            [0, 0, 0, 1]
        ])
        self.obs_pub = rospy.Publisher("rl_observation", Float32MultiArray, queue_size=10)
        rospy.Subscriber(self.markers_topic, MarkerArray, self.estimate_com_callback)

    def run(self):
        # run pose publisher
        rospy.Timer(rospy.Duration(0.05),
                    lambda msg: self.publisherCallback(msg))
    
    # def rgb_callback(self, msg):
    #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    #     self.rgb_image = np.asarray(cv_image)
    
    def publisherCallback(self, msg):
        if self.eef_pos is not None and self.obj1_pos is not None and self.obj2_pos is not None:
            if self.offset is None:
                self.offset = np.array([-0.35, 0.5, 0.42 - min(self.obj1_pos[2], self.obj2_pos[2])])
                obs_obj1_pos = np.array([self.obj1_pos[1], -self.obj1_pos[0], self.obj1_pos[2]]) + self.offset
            
                # obs_obj2_pos = np.array([self.obj2_pos[1], -self.obj2_pos[0], self.obj2_pos[2]]) + self.offset
            
                # self.goal1 = obs_obj1_pos + np.array([-0.2 * 0, 0.0, 0.0])
            # observation = np.concatenate(
            #     [self.obj_pos, self.eef_pos, self.eef_quaternion, self.finger_joints, 
            #      self.target_pos, self.goal])
            obs_eef_pos = np.array([self.eef_pos[1], -self.eef_pos[0], self.eef_pos[2]]) + self.offset
            obs_obj1_pos = np.array([self.obj1_pos[1], -self.obj1_pos[0], self.obj1_pos[2]]) + self.offset
            obs_obj2_pos = np.array([self.obj2_pos[1], -self.obj2_pos[0], self.obj2_pos[2]]) + self.offset
            observation = np.concatenate([obs_eef_pos, np.zeros(3), self.eef_vel, self.eef_vel, 
                                          [np.sum(self.finger_joints) / 2], [0], 
                                          np.array([0, 0, 0, 1]), obs_obj1_pos, 
                                          np.array([0, 0, 0, 1]), obs_obj2_pos,
                                          self.goal1, self.goal2])
            observation[:3] = (observation[:3]-self.goal_mean)/self.goal_std
            # observation[3:6] = (observation[3:6]-self.goal_mean)/self.goal_std
            observation[12:13] = (observation[12:13] - 0.02) / 0.011547
            observation[18:21] = (observation[18:21]-self.goal_mean)/self.goal_std
            observation[25:28] = (observation[25:28]-self.goal_mean)/self.goal_std
            observation[28:31] = (observation[28:31]-self.goal_mean)/self.goal_std
            observation[31:34] = (observation[31:34]-self.goal_mean)/self.goal_std
            print(observation)
            observation = Float32MultiArray(data=observation)
            # print("publish", observation)
            # saved_data = {"image": self.rgb_image, "q": self.robot_q, "eef_pos": self.eef_pos, 
            #           "finger_width": self.finger_joints, "box": self.obj_pos, "observation": observation}
            # with open("/home/yunfei/Documents/real_data/%d.pkl" % self.step_count, "wb") as f:
            #     pickle.dump(saved_data, f)
            self.step_count += 1
            self.obs_pub.publish(observation)
        else:
            print("observation not ready")
            print("obj_pos", self.obj1_pos, self.obj2_pos, "eef_pos", self.eef_pos)

    def franka_state_callback(self, msg):
        O_T_EE = np.transpose(np.reshape(msg.O_T_EE, (4, 4)))
        eef_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
        self.eef_quaternion = eef_quaternion / np.linalg.norm(eef_quaternion)
        self.eef_pos = np.array(tf.transformations.translation_from_matrix(O_T_EE))
        self.robot_q = np.array(msg.q)
        if not self._initial_state_found:
            self._initial_state_found = True
    
    def franka_gripper_state_callback(self, msg):
        self.finger_joints[0] = msg.position[0]
        self.finger_joints[1] = msg.position[1]
        if not self._initial_finger_found:
            self._initial_finger_found = True

    def estimate_com_callback(self, msg):
        # print("in estimate_com_callback")
        markers = msg.markers
        ref_frame = msg.header.frame_id
        ref_tvec, ref_rvec = self.tf_listener.lookupTransform(self.link_name, ref_frame, rospy.Time())
        O_T_ref = tf.transformations.quaternion_matrix(ref_rvec)
        O_T_ref[:3, 3] = ref_tvec
        # assume single object
        com_obs_1 = []
        com_obs_2 = []
        for i in range(len(markers)):
            marker_id = markers[i].id  # useful to filter when tracking multiple objects
            pose = markers[i].pose.pose
            tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
            rvec = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            ref_T_marker = tf.transformations.quaternion_matrix(rvec)
            ref_T_marker[:3, 3] = tvec
            if marker_id in self.marker_right_id:
                O_T_center = np.matmul(np.matmul(O_T_ref, ref_T_marker), self.mr_T_center)
            elif marker_id in self.marker_left_id:
                O_T_center = np.matmul(np.matmul(O_T_ref, ref_T_marker), self.ml_T_center)
            else:
                O_T_center = np.matmul(np.matmul(O_T_ref, ref_T_marker), self.m_T_center)
            if marker_id in self.marker1_id:
                print(marker_id, O_T_center[:3, 3])
                com_obs_1.append(O_T_center[:3, 3] + np.array([-0.015 * 0, 0.0, 0.0]))
            elif marker_id in self.marker2_id:
                com_obs_2.append(O_T_center[:3, 3] + np.array([0.0, 0.0, 0.0]))
        if len(com_obs_1):
            self.obj1_pos_obs.append(np.mean(np.array(com_obs_1), axis=0))
            self.obj1_pos = np.mean(np.array(self.obj1_pos_obs), axis=0)
        if len(com_obs_2):
            self.obj2_pos_obs.append(np.mean(np.array(com_obs_2), axis=0))
            self.obj2_pos = np.mean(np.array(self.obj2_pos_obs), axis=0)


if __name__ == "__main__":
    rospy.init_node("obs_publisher_node")
    link_name = rospy.get_param("~link_name")
    markers_topic = rospy.get_param("~markers_topic")
    obs_publisher = StateObsPublisher(link_name, markers_topic)
    obs_publisher.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")

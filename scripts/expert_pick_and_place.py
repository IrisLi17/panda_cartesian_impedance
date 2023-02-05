#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np
import math
import actionlib
from collections import deque
import tf2_ros
from std_msgs.msg import Int32

from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from franka_msgs.msg import FrankaState
import franka_gripper
import franka_gripper.msg
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from aruco_msgs.msg import MarkerArray
from cv_bridge import CvBridge


class ExpertController(object):
    def __init__(self):
        self.num_obj = 3
        self.obj_tags = np.array([0, 1, 3, 4])
        self.goal_tags = np.array([10, 2, 31, 14])
        self.grasp_disp = np.array([0.01, 0.075, 0.005])
        self.goal_disp = np.array([0.06, 0.0, 0.005])
        self.lift_height = 0.15
        self.reach_thereshold = 0.08
        self.control_err = 0.02

        # handover related virtual objects
        self.forward_handover_obj_ids = np.array([])
        self.backward_handover_obj_ids = np.array([])
        self.virtual_g_pos = np.array([0.5, -0.6, 0.02])  
        self.virtual_obj_pos =  np.array([0.5, -0.6, 0.04])

        self.obj_width = 0.04
        self.gripper_force = 0.1
        self.min_height = 0.03
        self.max_move_per_step = 0.08

        # handover state publisher
        self.current_forward_obj_id = -1
        self.forward_handover_state = 0 # 0: not started, 1: started, 2: ready, 3: finished
        self.backward_handover_state = 0 # 0: not started, 1: started, 2: ready, 3: finished
        self.forward_obj_id_pub = rospy.Publisher(
            "right_arm/forward_obj_id", Int32, queue_size=10)
        self.forward_handover_state_pub = rospy.Publisher(
            "right_arm/forward_handover", Int32, queue_size=10)
        self.backward_handover_state_pub = rospy.Publisher(
            "right_arm/backward_handover", Int32, queue_size=10)
        # handover state subscriber
        self.current_backward_obj_id = -1
        self.other_forward_handover_state = 0
        self.other_backward_handover_state = 0
        self.forward_obj_id_sub = rospy.Subscriber(
            "left_arm/forward_obj_id", Int32, self.forward_obj_id_callback)
        self.forward_handover_state_sub = rospy.Subscriber(
            "left_arm/forward_handover", Int32, self.forward_handover_state_callback)
        self.backward_handover_state_sub = rospy.Subscriber(
            "left_arm/backward_handover", Int32, self.backward_handover_state_callback)

        self.obj_pos = np.zeros((self.num_obj, 3))
        self.g_pos = np.zeros((self.num_obj, 3))
        self.obj_angle = np.zeros((self.num_obj, 1))
        self.obj_pos_his, self.goal_pos_his, self.obj_angle_his = [], [], []
        self.unobserved_obj_t = np.zeros((self.num_obj))
        self.unobserved_g_t = np.zeros((self.num_obj))
        self.unobserved_thereshold = 20
        # NOTE: using 
        for _ in range(self.num_obj):
            self.obj_pos_his.append(deque(maxlen=10))
            self.goal_pos_his.append(deque(maxlen=10))
            self.obj_angle_his.append(deque(maxlen=10))

        self.current_obj_id = 0
        self.target_obj_pos = np.zeros(3)
        self.target_obj_angle = 0
        # create object rotating matrix from object angle
        self.target_obj_rot_mat = np.array([[np.cos(self.target_obj_angle), -np.sin(self.target_obj_angle), 0],
                                        [np.sin(self.target_obj_angle), np.cos(self.target_obj_angle), 0],
                                        [0, 0, 1]])
        self.target_goal_pos = np.zeros(3)


        # create tf2 boardcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.desired_pose = PoseStamped()
        self._initial_pose_found = False
        self.eef_pos = np.zeros(3)
        self.robot_q = None
        self.target_obj_pos_obs = deque(maxlen=10)
        self.goal_pos_obs = deque(maxlen=10)
        self.finger_width = 0.08
        self.step_count = 0
        state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states",
            FrankaState, self.franka_state_callback)
        rospy.Subscriber("franka_gripper/joint_states", 
                    JointState, self.franka_gripper_state_callback)
        self.tf_listener = tf.TransformListener()
        # robot base frame
        self.link_name = rospy.get_param("~link_name")
        self.markers_topic = rospy.get_param("~markers_topic") # /marker_publisher/markers
        rospy.Subscriber(self.markers_topic, MarkerArray, self.estimate_com_callback)
        # self.rgb_image = None
        # self.bridge = CvBridge()
        # rospy.Subscriber("camera/color/image_raw", Image, self.rgb_callback)
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
        self.phase = -2 # for control
        self.gripper_grasp_lock = 0
        self.gripper_open_lock = 0

        self.obs_history = []
        # run pose publisher
        self.timer = rospy.Timer(rospy.Duration(0.1),
                    lambda msg: self.control_callback(msg))

        self.start_lock = True
    
    def forward_obj_id_callback(self, msg):
        self.current_backward_obj_id = msg.data

    def forward_handover_state_callback(self, msg):
        self.other_forward_handover_state = msg.data

    def backward_handover_state_callback(self, msg):
        self.other_backward_handover_state = msg.data
    
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
            self.robot_q = np.array(msg.q)

    
    def franka_gripper_state_callback(self, msg):
        self.finger_width = msg.position[0] + msg.position[1]
    
    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.rgb_image = np.asarray(cv_image)

    def estimate_com_callback(self, msg):
        # print("in estimate_com_callback")
        markers = msg.markers
        ref_frame = msg.header.frame_id
        ref_tvec, ref_rvec = self.tf_listener.lookupTransform(self.link_name, ref_frame, rospy.Time())
        O_T_ref = tf.transformations.quaternion_matrix(ref_rvec)
        O_T_ref[:3, 3] = ref_tvec
        # assume single object
        com_obs = []
        # compute unobserved object time
        self.unobserved_obj_t += 1
        self.unobserved_g_t += 1
        for i in range(len(markers)):
            marker_id = markers[i].id  # useful to filter when tracking multiple objects
            # publish marker to tf tree with tf2
            pose = markers[i].pose.pose
            # directly publish the pose to tf tree
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = ref_frame
            t.child_frame_id = "marker_" + str(marker_id)
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation.x = pose.orientation.x
            t.transform.rotation.y = pose.orientation.y
            t.transform.rotation.z = pose.orientation.z
            t.transform.rotation.w = pose.orientation.w
            self.tf_broadcaster.sendTransform(t)
            if marker_id in self.obj_tags:
                # get marker_id index in self.obj_tags
                obj_id = np.where(self.obj_tags == marker_id)[0][0]
                self.unobserved_obj_t[obj_id] = 0
                ref_tvec, ref_rvec = self.tf_listener.lookupTransform(self.link_name, t.child_frame_id, rospy.Time())
                # get euler angle from quaternion
                euler_angle = tf.transformations.euler_from_quaternion(ref_rvec)
                self.obj_pos_his[obj_id].append(ref_tvec)
                self.obj_angle_his[obj_id].append(euler_angle[2])
            if marker_id in self.goal_tags:
                # get marker_id index in self.obj_tags
                goal_id = np.where(self.goal_tags == marker_id)[0][0]
                self.unobserved_g_t[goal_id] = 0
                ref_tvec, ref_rvec = self.tf_listener.lookupTransform(self.link_name, t.child_frame_id, rospy.Time())
                self.goal_pos_his[goal_id].append(ref_tvec)
        observed_obj_mask = self.unobserved_obj_t < self.unobserved_thereshold
        observed_g_mask = self.unobserved_g_t < self.unobserved_thereshold
        for i in np.arange(self.num_obj)[observed_obj_mask]:
            # set self.obj_pos to the mean of self.obj_pos_his
            self.obj_pos[i] = np.mean(np.asarray(self.obj_pos_his[i]), axis=0)
            self.obj_angle[i] = np.mean(np.asarray(self.obj_angle_his[i]), axis=0)
        for i in np.arange(self.num_obj)[observed_g_mask]:
            self.g_pos[i] = np.mean(np.asarray(self.goal_pos_his[i]), axis=0) + self.goal_disp
            self.g_pos[i][2] = 0.02
        for i in np.arange(self.num_obj)[~observed_obj_mask]:
            # relabel to virtual object pos
            self.obj_pos[i] = self.virtual_obj_pos
            self.obj_angle[i] = 0
        for i in np.arange(self.num_obj)[~observed_g_mask]:
            self.g_pos[i] = self.virtual_g_pos
        # self.backward_handover_obj_ids = np.where(observed_g_mask&(~observed_obj_mask))[0]
        # self.forward_handover_obj_ids = np.where(observed_obj_mask&(~observed_g_mask))[0]

    def control_callback(self, msg):
        if self.phase == -2:
            if self.start_lock:
                # wait util all observation is received
                rospy.sleep(20)
                self.start_lock = False
            observed_obj_mask = self.unobserved_obj_t < self.unobserved_thereshold
            observed_g_mask = self.unobserved_g_t < self.unobserved_thereshold
            self.forward_handover_obj_ids = np.where(observed_obj_mask&(~observed_g_mask))[0]
            self.backward_handover_obj_ids = np.where(observed_g_mask&(~observed_obj_mask))[0]
            if np.all(self.obj_pos == 0):
                print("Box pos not received")
            elif np.all(self.g_pos == 0):
                print("Goal pos not received")
            else:
                print(self.forward_handover_obj_ids, self.backward_handover_obj_ids)
                self.phase = -1
        
        elif self.phase == -1:
            # planning phase
            self.current_obj_id = self.get_next_obj_id()
            # overwrite current_obj_id if handover is needed
            if self.current_backward_obj_id >= 0 and self.other_forward_handover_state >0:
                self.current_obj_id = self.current_backward_obj_id
                print('override current_obj_id with other_forward_hand')
            elif self.current_obj_id in self.backward_handover_obj_ids:
                # if determined backward object id is not handovered back, then ignored. 
                print('backward handover object', self.current_obj_id, 'is not handovered back in', self.current_backward_obj_id)
                self.current_obj_id = -1
            # update handover status
            if self.current_obj_id in self.backward_handover_obj_ids:
                self.backward_handover_status = 1
            else:
                self.backward_handover_status = 0
            if self.current_obj_id in self.forward_handover_obj_ids:
                self.current_forward_obj_id = self.current_obj_id
                self.forward_handover_status = 1
            else:
                self.current_forward_obj_id = -1 
                self.forward_handover_status = 0
            if self.current_obj_id == -1:
                print("All objects are done")
                self.phase = -2
            else:
                self.target_obj_pos = self.obj_pos[self.current_obj_id].copy()
                self.target_obj_angle = self.obj_angle[self.current_obj_id]
                self.target_obj_rot_mat = np.array([[np.cos(self.target_obj_angle), -np.sin(self.target_obj_angle), 0],
                                                    [np.sin(self.target_obj_angle), np.cos(self.target_obj_angle), 0],
                                                    [0, 0, 1]])
                self.target_goal_pos = self.g_pos[self.current_obj_id].copy()
                self.phase = 0
        
        if self.phase == 0:
            target_pose = self.target_obj_pos+ np.matmul(self.target_obj_rot_mat, self.grasp_disp) + np.array([0, 0, self.lift_height])
            if self.backward_handover_status == 1:
                target_pose = target_pose + np.array([0, 0, self.lift_height])
            # pregrasp phase
            dpos = np.clip(target_pose - self.eef_pos, -0.05 ,0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            angle = (np.pi/2-self.target_obj_angle)/2
            self.desired_pose.pose.orientation.x = np.sin(angle)
            self.desired_pose.pose.orientation.y = np.cos(angle)
            self.desired_pose.pose.orientation.z = 0.0
            self.desired_pose.pose.orientation.w = 0.0
            
            # open gripper
            if not self.gripper_open_lock:
                goal = franka_gripper.msg.MoveGoal(width=0.08, speed=0.1)
                self.gripper_move_client.send_goal(goal)
                self.gripper_open_lock = 1
            if self.gripper_open_lock and self.finger_width > 0.078:
                self.gripper_move_client.cancel_all_goals()
                self.gripper_open_lock = 0

            print("In phase", self.phase, "hand pos", self.eef_pos, "error", np.linalg.norm(dpos))
            if np.linalg.norm(dpos) < self.control_err:
                if self.current_obj_id in self.backward_handover_obj_ids:
                    if self.other_forward_handover_state == 2: 
                        self.phase = 1
                else:
                    self.phase = 1
        elif self.phase == 1:
            target_pose = self.target_obj_pos+ np.matmul(self.target_obj_rot_mat, self.grasp_disp)
            if self.backward_handover_status == 1:
                target_pose = target_pose + np.array([0, 0, self.lift_height])
            dpos = np.clip(target_pose - self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            print("In phase", self.phase, "hand pos", self.eef_pos, "error", np.linalg.norm(dpos))
            if np.linalg.norm(dpos) < self.control_err:
                self.phase = 2
        elif self.phase == 2:
            # close gripper phase
            if not self.gripper_grasp_lock:
                epsilon = franka_gripper.msg.GraspEpsilon(inner=0.02, outer=0.02)
                goal = franka_gripper.msg.GraspGoal(
                    width=self.obj_width, speed=0.1, epsilon=epsilon, force=self.gripper_force)

                self.gripper_grasp_client.send_goal(goal)
                self.gripper_grasp_lock = 1
                self.gripper_grasp_client.wait_for_result(rospy.Duration(0.1))
            print("In phase", self.phase, self.finger_width)
            if self.gripper_grasp_lock and self.finger_width < (self.obj_width+0.005):
                self.gripper_grasp_lock = 0
                self.up_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] + self.lift_height]
                if self.current_obj_id in self.backward_handover_obj_ids:
                    self.backward_handover_state = 2
                    if self.other_forward_handover_state == 0:
                        self.phase = 3
                    else:
                        self.gripper_grasp_lock = 1
                        self.phase = 2
                else:
                    self.phase = 3
        elif self.phase == 3:
            # lift phase
            target_pose = self.up_pos
            dpos = np.clip(target_pose- self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            print("In phase", self.phase, "hand pos", self.eef_pos)
            if abs(self.eef_pos[2] - self.up_pos[2]) < 3e-2:
                self.gripper_grasp_client.cancel_all_goals()
                self.phase = 4
        elif self.phase == 4:
            # preplace phase
            dpos = np.clip(self.target_goal_pos + self.grasp_disp + np.array([0, 0, self.lift_height]) - self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            angle = (np.pi/2+0.0)/2
            self.desired_pose.pose.orientation.x = np.sin(angle)
            self.desired_pose.pose.orientation.y = np.cos(angle)
            self.desired_pose.pose.orientation.z = 0.0
            self.desired_pose.pose.orientation.w = 0.0
            print("In replace", self.phase, "hand pos", 'goal', self.target_goal_pos, 'obj', self.target_obj_pos, self.eef_pos, "error", np.linalg.norm(dpos), 'forward', self.forward_handover_obj_ids)
            if np.linalg.norm(dpos) < self.control_err:
                if self.current_obj_id in self.forward_handover_obj_ids:
                    # update handover state
                    self.forward_handover_state = 2
                    # wait util other arm is handover ready
                    if self.other_backward_handover_state==2:
                        self.phase = 6
                    else:
                        self.phase = 4
                else:
                    self.phase = 5
                self.gripper_grasp_client.cancel_all_goals()
        elif self.phase == 5:
            # place phase
            target_pose = self.target_goal_pos + self.grasp_disp 
            target_pose[2] = np.clip(target_pose[2], self.min_height, 10)
            dpos = np.clip(target_pose- self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            print("In replace", self.phase, "hand pos", self.eef_pos, "goal", self.target_goal_pos, "error", np.linalg.norm(dpos))
            if np.linalg.norm(dpos) < self.control_err:
                self.phase = 6
        elif self.phase == 6:
            # open gripper
            if not self.gripper_open_lock:
                goal = franka_gripper.msg.MoveGoal(width=0.08, speed=0.1)
                self.gripper_move_client.send_goal(goal)
                self.gripper_open_lock = 1
            if self.gripper_open_lock and self.finger_width > 0.078:
                self.phase = 7
                self.gripper_move_client.cancel_all_goals()
                self.gripper_open_lock = 0
                if self.forward_handover_state == 2:
                    self.up_pos = [self.eef_pos[0], self.eef_pos[1] + self.lift_height*2.0, self.eef_pos[2]]
                else:
                    self.up_pos = [self.eef_pos[0], self.eef_pos[1], self.eef_pos[2] + self.lift_height*2.0]
        elif self.phase == 7:
            # lift phase
            target_pose = self.up_pos
            dpos = np.clip(target_pose- self.eef_pos, -0.05, 0.05)
            self.desired_pose.pose.position.x = self.eef_pos[0] + dpos[0]
            self.desired_pose.pose.position.y = self.eef_pos[1] + dpos[1]
            self.desired_pose.pose.position.z = self.eef_pos[2] + dpos[2]
            print("In phase", self.phase, "hand pos", self.eef_pos)
            if abs(self.eef_pos[2] - self.up_pos[2]) < 3e-2:
                if self.current_obj_id in self.forward_handover_obj_ids:
                    self.forward_handover_state = 0
                self.gripper_grasp_client.cancel_all_goals()
                self.phase = -2
                # clean the state variables for next stage observation
                self.obj_pos = np.zeros((self.num_obj, 3))
                self.g_pos = np.zeros((self.num_obj, 3))
        self.pose_pub.publish(self.desired_pose)
        self.backward_handover_state_pub.publish(self.backward_handover_state)
        self.forward_handover_state_pub.publish(self.forward_handover_state)

    def get_next_obj_id(self):
        obj2goal_dist = np.linalg.norm((self.obj_pos - self.g_pos)[:, :2], axis=1)
        unreached_objects = np.where(obj2goal_dist > self.reach_thereshold)[0]
        if len(unreached_objects) == 0:
            return -1
        else: 
            # return the object with the smallest distance to the goal
            return unreached_objects[np.argmin(obj2goal_dist[unreached_objects])]


if __name__ == "__main__":
    rospy.init_node("pose_commander_node")
    controller = ExpertController()
    rospy.spin()
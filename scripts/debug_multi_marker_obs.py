#!/usr/bin/env python

import rospy
import tf.transformations
from aruco_msgs.msg import MarkerArray
import numpy as np


class Test():
    def __init__(self):
        self.markers_topic = rospy.get_param("~markers_topic") # /marker_publisher/markers
        rospy.Subscriber(self.markers_topic, MarkerArray, self.estimate_com_callback)
    
    def estimate_com_callback(self, msg):
        print("in estimate_com_callback")
        markers = msg.markers
        ref_frame = msg.header.frame_id
        # assume single object
        com_obs = []
        for i in range(len(markers)):
            marker_id = markers[i].id  # useful to filter when tracking multiple objects
            pose = markers[i].pose.pose
            tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
            rvec = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            ref_T_marker = tf.transformations.quaternion_matrix(rvec)
            ref_T_marker[:3, 3] = tvec
            com_obs.append(ref_T_marker[:3, 3])
        self.box_pos = np.mean(np.array(com_obs), axis=0)
        print("box pos", self.box_pos)
    

if __name__ == "__main__":
    rospy.init_node("debug_node")
    controller = Test()
    rospy.spin()
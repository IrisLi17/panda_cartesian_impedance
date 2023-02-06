#!/usr/bin/env python  
import rospy

from std_msgs.msg import Int32

if __name__ == '__main__':
    rospy.init_node('right_arm')

    forward_handover_state_pub = rospy.Publisher(
            "right_arm/forward_handover", Int32, queue_size=10)
    backward_handover_state_pub = rospy.Publisher(
        "right_arm/backward_handover", Int32, queue_size=10)
    forward_obj_id_pub = rospy.Publisher(
        "right_arm/forward_obj_id", Int32, queue_size=10)

    while not rospy.is_shutdown():
        fb = input('right forward state, backward state, forward obj:')
        forward_handover_state_pub.publish(int(fb[0]))
        backward_handover_state_pub.publish(int(fb[1]))
        forward_obj_id_pub.publish(int(fb[2]))
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(0)
        # backward_handover_state_pub.publish(0)
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(1)
        # backward_handover_state_pub.publish(1)
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(2)
        # backward_handover_state_pub.publish(2)


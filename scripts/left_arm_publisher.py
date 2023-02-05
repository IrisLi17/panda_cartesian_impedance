#!/usr/bin/env python  
import rospy

from std_msgs.msg import Int32

if __name__ == '__main__':
    rospy.init_node('left_arm')

    forward_handover_state_pub = rospy.Publisher(
            "left_arm/forward_handover", Int32, queue_size=10)
    backward_handover_state_pub = rospy.Publisher(
        "left_arm/backward_handover", Int32, queue_size=10)

    while not rospy.is_shutdown():
        fb = input('left forward state, backward state:')
        forward_handover_state_pub.publish(int(fb[0]))
        backward_handover_state_pub.publish(int(fb[1]))
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(0)
        # backward_handover_state_pub.publish(0)
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(1)
        # backward_handover_state_pub.publish(1)
        # rospy.sleep(3.0)
        # forward_handover_state_pub.publish(2)
        # backward_handover_state_pub.publish(2)


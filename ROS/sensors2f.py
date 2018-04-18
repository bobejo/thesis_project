#!/usr/bin/env python

from functools import partial

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, WrenchStamped
from robotiq_c_model_control.msg import _CModel_robot_input as gripperIn

class Sensors:
    def __init__(self):
        # States-------------------------------------------------------------------------

        # robot
        self.joint_states = [JointState()]
        self.tool_velocity = [TwistStamped()]
        self.internal_wrench = [WrenchStamped()]

        # robotiq force sensor
        self.external_wrench = [WrenchStamped()]

        # robotiq gripper
        self.gripper = [gripperIn.CModel_robot_input()]

        # Subscribers-----------------------------------------------------------------

        # robot
        rospy.Subscriber("/joint_states", JointState, partial(self.updateState_cb, self.joint_states))
        rospy.Subscriber("/tool_velocity", TwistStamped, partial(self.updateState_cb, self.tool_velocity))
        rospy.Subscriber("/wrench", WrenchStamped, partial(self.updateState_cb, self.internal_wrench))

        # robotiq force sensor
        rospy.Subscriber("/robotiq_force_torque_wrench", WrenchStamped, partial(self.updateState_cb, self.external_wrench))

        # robotiq gripper
        rospy.Subscriber("/CModelRobotInput", gripperIn.CModel_robot_input, partial(self.updateState_cb, self.gripper))

    # CBs ----------------------------------------------
    def updateState_cb(self, stateToUpdate, msg):
        stateToUpdate[0] = msg




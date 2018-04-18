#!/usr/bin/env python

import rospy

from robotiq_c_model_control.msg import _CModel_robot_output  as outputMsg


class CModelGripper:
    def __init__(self):
        # Publishers---------------
        # robotiq gripper
        self.gripperPub = rospy.Publisher('CModelRobotOutput', outputMsg.CModel_robot_output, queue_size=1)

        # Variables----------------
        self.command = outputMsg.CModel_robot_output()

    def publish(self):
        self.gripperPub.publish(self.command)
        rospy.sleep(0.2)

    def resetGripper(self):
        self.command = outputMsg.CModel_robot_output();
        self.command.rACT = 0
        self.publish()

    def activateGripper(self):
        self.command = outputMsg.CModel_robot_output();
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rSP = 255
        self.command.rFR = 150
        self.publish()

    def positionGripper(self, pos):
        self.command.rPR = int(pos)
        if self.command.rPR > 255:
            self.command.rPR = 255
        if self.command.rPR < 0:
            self.command.rPR = 0
        self.publish()

    def openGripper(self):
        self.positionGripper(0)

    def closeGripper(self):
        self.positionGripper(255)

    def setGripperSpeed(self, speed):
        self.command.rSP = int(speed)
        if self.command.rSP > 255:
            self.command.rSP = 255
        if self.command.rSP < 0:
            self.command.rSP = 0
        self.publish()

    def setGripperForce(self, force):
        self.command.rFR = int(force)
        if self.command.rFR > 255:
            self.command.rFR = 255
        if self.command.rFR < 0:
            self.command.rFR = 0
        self.publish()

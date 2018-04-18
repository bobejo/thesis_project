#!/usr/bin/env python
"""
Setting different modes of operation for Robotic 3_finger_S_Model_Gripper
"""
import rospy

from robotiq_s_model_control.msg import _SModel_robot_output  as outputMsg

class SModelGripper:
    def __init__(self):
        # Publishers---------------
        # robotiq gripper
        self.gripperPub = rospy.Publisher('SModelRobotOutput', outputMsg.SModel_robot_output, queue_size=1)
        
        # Variables----------------
        self.command = outputMsg.SModel_robot_output()

    def publish(self):
        self.gripperPub.publish(self.command)
        rospy.sleep(0.2)

    def resetGripper(self):
        self.command = outputMsg.SModel_robot_output();
        self.command.rACT = 0
        self.publish()

    def activateGripper(self):
        self.command = outputMsg.SModel_robot_output(); 
        # Indvidual finger control (active rICF=1, deactive rICF=0)
        self.command.rICF = 1   
        # Operating in scessor mode (active rICS=1, deactive rICS=0)
        self.command.rICS= 0   
        self.command.rSPS= 255
        self.command.rFRS= 150  

        self.command.rACT = 1
        self.command.rGTO = 1
        # Finger C
        self.command.rSPC  = 255
        self.command.rFRC  = 150
        # Finger B
        self.command.rSPB  = 255
        self.command.rFRB  = 150
        # Finger A
        self.command.rSPA = 255
        self.command.rFRA = 150
        self.publish()



    # Operating in Base Mode
    def openGripperBase(self):
        self.command.rPRA= 0
        self.publish()

    def closeGripperBase(self):
        self.command.rPRA= 250
        self.publish()

    def setGripperSpeedBase(self):
        self.command.rSPA = 100
        self.publish()

    def setGripperForceBase(self):
        self.command.rFRA = 100
        self.publish()



    # Operating in Scissor Mode (set rCIS to 1)
    def openGripperScissor(self):
        # Control of fingers B and C (Scissor Mode)
        self.command.rPRS= 0
        self.publish()

    def closeGripperScissor(self):
        self.command.rPRS= 250
        self.publish()

    def setGripperSpeedScissor(self):
        self.command.rSPS = 100
        self.publish()

    def setGripperForceScissor(self):
        self.command.rFRS = 100
        self.publish()


    # Indvidual fingers (set rCIF to 1)
    def openGripperInd(self):
        # Control the desired fingure
        # self.command.rPRC = 0 #(C)
        # self.command.rPRB = 0 #(B)
        self.command.rPRA = 0 #(A- Middle)
        self.publish()

    def closeGripperInd(self):
        # self.command.rPRB = 255
        # self.command.rPRC = 255
        self.command.rPRA = 255
        self.publish()

    def setGripperSpeedInd(self):
        # self.command.rSPC = 100
        # self.command.rSPB = 100
        self.command.rSPA = 100
        self.publish()

    def setGripperForceInd(self):
        # self.command.rFRC = 100
        # self.command.rFRB = 100
        self.command.rFRA = 100
        self.publish()










    # def setGripperSpeed(self, speed):
    #     self.command.rICF = 1
    #     self.command.rSPB = int(speed)
    #     if self.command.rSPB > 255:
    #         self.command.rSPB = 255
    #     if self.command.rSPB < 0:
    #         self.command.rSPB = 0
    #     self.publish()

    # def setGripperForce(self, force):
    #     self.command.rICF = 1
    #     self.command.rFRB = int(force)
    #     if self.command.rFRB > 255:
    #         self.command.rFRB = 255
    #     if self.command.rFRB < 0:
    #         self.command.rFRB = 0
    #     self.publish()


#!/usr/bin/env python

#import json

import rospy, roslib
from std_msgs.msg import String


from support3f import Support3
from sensors3f import Sensors3
from sModelGripper import SModelGripper
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates
from geometry_msgs.msg import WrenchStamped
from robotiq_s_model_control.msg import _SModel_robot_input  as inputMsg

# from SetIO.srv import *
from ur_msgs.srv import SetIO
#for socket program
import socket
import time
import pigpio

import numpy
import rospy
import time
import datetime


#For Gripper
import roslib; roslib.load_manifest('robotiq_s_model_control')
import rospy
from robotiq_s_model_control.msg import _SModel_robot_output  as outputMsg
from time import sleep
hz=10

class Picking(Support3, Sensors3, SModelGripper):
	def __init__(self):
		rospy.init_node('Picking')
		Support3.__init__(self)
		Sensors3.__init__(self)
		SModelGripper.__init__(self)

		# rospy.Subscriber("/joint_states", JointState , self.jointCallback) 
		# rospy.Subscriber("/optoforce_0", WrenchStamped , self.callback_tool)
        #rospy.Subscriber("/CModelRobotInput", inputMsg.CModel_robot_input, self.gripperCallback)
		#rospy.Subscriber("/optoforce_0", WrenchStamped, self.callback_tool)

		#Robot Publisher
		# self.MovePublisher = rospy.Publisher("/ur_driver/URScript", String, queue_size=1)
		
		self.rate = rospy.Rate(10)  # 10hz
		rospy.loginfo(rospy.get_caller_id() + "Test started...")
		time.sleep(2)

		# Initialize Gripper
		self.resetGripper()
		self.activateGripper()
		# self.positionGripper(100) 
		self.setGripperForceInd()
		self.setGripperSpeedInd()
		time.sleep(10)
		self.setGripperSpeedInd()
		self.closeGripperInd()
		rospy.sleep(10)
		self.openGripperInd()

		# Move to specific point in xyz
		# command = "movel(p[-0.543,0.106,0.199,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(7)

		# command = "movel(p[-0.5449,-0.123,0.196,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.541,0.106,0.216,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.543,-0.123,0.222,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.498,-0.203,0.098,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.477,-0.203,0.092,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.542,-0.0096,0.222,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)

		# command = "movel(p[-0.661,0.012,0.149,2.40,-1.56,-0.36],a=0.1, v=0.1)"
		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)




		# print self.command
		# self.MovePublisher.publish(command)
		# rospy.sleep(3)
		# self.setGripperSpeed(50)
		# self.closeGripper()
		# rospy.sleep(3)
		# self.openGripper()




		# Move to required position

		# self.startPosition=[4.529989719390869, -1.3746760527240198, -1.9332359472857874, 4.5790886878967285, 1.4938651323318481, 2.9489781856536865]
		# self.startPosition=p[-0.243,-0.434,0.288,3.2890,-0.1253,-0.0127]
		# command = self.urSrciptToString(
		# 		move="movel",
		# 		jointPose=self.startPosition,
		# 		a=0.001,
		# 		v=0.001,
		# 		t=5,
		# 		r=0)
		# print command    
		# self.MovePublisher.publish(command)
		# # time.sleep(5)
		# # self.setGripperSpeed(50)
		# # self.closeGripper()
		# # print "Gripper Close"
		# # time.sleep(3)
		# # self.openGripper()
		# # print "Gripper open"


		self.rate10 = rospy.Rate(10)          # 10hz
		self.rate100 = rospy.Rate(100)          # 100hz
		rospy.loginfo(rospy.get_caller_id() + "Test started...")


	# def jointCallback(self,position):
	# 	pass





	# def ft_measure(self,data):
	# 	# Get Forces Measurements
	# 	self.force_x=float(data.wrench.force.x)
	# 	self.force_y=float(data.wrench.force.y)
	# 	self.force_z=float(data.wrench.force.z)

	# 	# Get Torques Measurements
	# 	self.torque_x=float(data.wrench.torque.x)
	# 	self.torque_y=float(data.wrench.torque.y)
	# 	self.torque_z=float(data.wrench.torque.z)



	# def callback_tool(self, data):
	# 	self.force_x = float(data.wrench.force.x)
	# 	self.force_y = float(data.wrench.force.y)
	# 	self.force_z = float(data.wrench.force.z)
	# 	self.torque_x = float(data.wrench.torque.x)
	# 	self.torque_y = float(data.wrench.torque.y)
	# 	self.torque_z = float(data.wrench.torque.z)
 #        #  print self.force_z








if __name__ == '__main__':
	try:
		Picking()
	except rospy.ROSInterruptException:
		pass
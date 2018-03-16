#!/usr/bin/env python

#import json
import rospy, roslib
from std_msgs.msg import String



# For snapping pictures
import os
import wget



#For Gripper
from support import Support
from sensors import Sensors
from cModelGripper import CModelGripper
from sensor_msgs.msg import JointState
from ur_msgs.msg import IOStates
from geometry_msgs.msg import WrenchStamped
from robotiq_c_model_control.msg import _CModel_robot_input  as inputMsg



# For Stereo Base transformation
# from stereotobase import StereoBase
import time
import tf
from numpy import matrix
from numpy import linalg
import math
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from numpy.linalg import inv



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
import roslib; roslib.load_manifest('robotiq_c_model_control')
import rospy
from robotiq_c_model_control.msg import _CModel_robot_output  as outputMsg
from time import sleep
hz=10







class Picking(Support, Sensors, CModelGripper):

	def __init__(self):

		rospy.init_node('Picking')
		Support.__init__(self)
		Sensors.__init__(self)
		CModelGripper.__init__(self)


		# Subscribe to Sensors (Force/Torque and Joints states)
		rospy.Subscriber("/joint_states", JointState , self.jointCallback) 
		rospy.Subscriber("/optoforce_0", WrenchStamped , self.callback_tool)

		# Gets and listens to any updates from the broadcaster 
		self.listener = tf.TransformListener()

        # Subscribe to robot
		rospy.Subscriber("/joint_states", JointState, self.cb1)

		#Publish to Robot
		self.MovePublisher = rospy.Publisher("/ur_driver/URScript", String, queue_size=1)
		
		self.rate = rospy.Rate(10)  # 10hz
		rospy.loginfo(rospy.get_caller_id() + "Test started...")
		rospy.sleep(2)

		# Display initial readings form F/T sensor
		self.measure_initial_states()

		# Initialize Gripper
		self.resetGripper()
		self.activateGripper()
		self.positionGripper(50) 
		self.setGripperForce(175)
		self.setGripperSpeed(100)
		rospy.sleep(2)

		# Snap a picture first
		self.snap_pic()
		rospy.sleep(2)

		# Move to a initial position (bin position)
		self.initial_position(0.01,0.01,5,0) # Given a, v,t, and r







	def snap_pic(self):
	    #Ensure that the counter is at the right number so there will be no duplicates of the image names
	    try:
	        f = open('counter', 'r')
	        counter = f.read()
	    except IOError:
	        f = open('counter', 'w+')
	        counter = 1
	    f.close()

	    print("Counter number", counter)
	    'http://192.168.1.138/dms?nowprofileid=1 -O'
	    os.system('wget -q http://admin:admin@192.168.1.138/dms?nowprofileid=1 -O' "image/left" + ".jpg")
	    print("Picture 1 left is saved")
	    os.system('wget -q http://admin:@192.168.1.144/dms?nowprofileid=1 -O' "image/right" + ".jpg")
	    print("Picture 1 right is saved")

	    counter = int(counter) + 1
	    f = open('counter', 'w')
	    f.write(str(counter))
	    f.close()





	def initial_position(self,a,v,t,r):
		self.startPosition=[3.1888632774353027, -1.4484122435199183, -1.947954003010885, -1.1429865995990198, -4.834929172192709, -0.19468576112856084]
		command = self.urSrciptToString(
			move="movej",
			jointPose=self.startPosition,
			a=0.01,
			v=0.01,
			t=5,
			r=0)
		print command 
		self.MovePublisher.publish(command)
		rospy.sleep(5)
		self.FTSframe()






# .............................................................. Transformation between Stereo Camera frame to Base frame ....................................................


	# def spin(self):
	# 	# while (not rospy.is_shutdown()):
	# 	start_time = rospy.get_rostime() 
	# 	end = time.time()

	# 	trans=self.FTSframe()

	# 		# Go into spin, with rateOption!
	# 	self.rate.sleep()  
  



	def cb1(self, data):
		# Get update from the manipulator:
		self.joints = data.position;  




	def FTSframe(self):
		self.listener.waitForTransform('/base','/stereo_camera', rospy.Time(0),rospy.Duration(1))
		(trans,rot) = self.listener.lookupTransform('/base','/stereo_camera', rospy.Time(0))

		# self.listener.waitForTransform('/stereo_camera','/base', rospy.Time(0),rospy.Duration(1))
		# (trans,rot) = self.listener.lookupTransform('/stereo_camera','/base', rospy.Time(0))


		transrotM = self.listener.fromTranslationRotation(trans, rot)   #  4x4 Transformation matrix
		print transrotM


		rotationMat = transrotM[0:3,0:3] # Rotation 
		translationMat= transrotM[:,3]   # Translation



		# point_t=(406,356,-176,1) #triangulation

		# The target position represented in base frame, can be realted to end-effector as well
		point_t=np.matrix([406,356,-176,1]) #triangulation
		point_tt= point_t.transpose()
		R_c_ee = np.dot(transrotM,point_tt)
		print R_c_ee
		self.grip(0.01,0.01,5,0)


		return transrotM
		return R_c_ee



			# rospy.sleep(3)
			# self.grip(0.01,0.01,5,0)

		# except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
		# 	print "EXCEPTION"
			# pass


# .........................................................................  End of Transformation  ...................................................................................

	# 	# self.list_start_position = [ '%.2f' % elem for elem in self.startPosition]  # Position we want to go
	# 	# 	# print self.list_start_position
	# 	# self.list_current_position = [ '%.2f' % elem for elem in self.current_position]  # Position of the robot, which is updated in real time



	# 	# while not rospy.is_shutdown():
	# 	# 	if self.list_start_position != self.list_current_position:
	# 	# 		print command 
	# 	# 		self.MovePublisher.publish(command)
	# 	# 		rospy.sleep(5)
	# 	# 		self.list_current_position = [ '%.2f' % elem for elem in self.current_position]

	# 	# 	else:
	# 	# 		break


	# 	# Make the frames transformations and get object position and orientation



	def grip(self,a,v,t,r):
		rospy.sleep(2)
		print "Move to Gripping position" # on the object
		rospy.sleep(2)

		# It should be the position of the object after transformation (movel p)
		self.gripPosition=[3.1888034343719482, -1.4483760038958948, -2.138037983571188, -1.1429508368121546, -4.834929172192709, -0.19472200075258428]
		grip = self.urSrciptToString(
			move="movej",
			jointPose=self.gripPosition,
			a=0.01,
			v=0.01,
			t=5,
			r=0)
		print grip 
		self.MovePublisher.publish(grip)
		rospy.sleep(5)

		print "Grip an object"
		rospy.sleep(5)
		self.setGripperSpeed(50)
		self.closeGripper()

		rospy.sleep(2)

		print "Current force in z= ", self.force_z
		print "Current torque in y= ", self.torque_y

		# Move up a bit with carried objects
		self.post_grip_position=[3.1888632774353027, -1.4484003225909632, -1.9196561018573206, -1.2954629103290003, -4.834917132054464, -0.19469768205751592]
		post_grip = self.urSrciptToString(
			move="movej",
			jointPose=self.post_grip_position,
			a=0.01,
			v=0.01,
			t=5,
			r=0)
		print post_grip 
		self.MovePublisher.publish(post_grip)
		rospy.sleep(5)
		self.check_picked_object()



	# Check the weight of picked object and perform the required action
	def check_picked_object(self):

		print "Current force in z= ", self.force_z
		print "Current torque in y= ", self.torque_y
		rospy.sleep(5)

		while not rospy.is_shutdown():
			if self.force_z<0:
				print "No object Picked"
				break

			if self.force_z>2.5 and self.force_z<7 and self.torque_y<0.008:
				print "A black tube is Picked"
				break

			elif self.force_z>3.5 and self.force_z<7 and self.force_z<10 and self.torque_y > 0.035:
				print "A U-tube is picked"
				break
			elif self.force_z>7:
				print "Warning!!!: Two U-tubes are picked"
				rospy.sleep(5)
				print grip 
				self.MovePublisher.publish(grip)
				rospy.sleep(5)
				break
		rospy.sleep(5)
		self.final_position(0.01,0.01,5,0) #table

		self.rate10 = rospy.Rate(10)          # 10hz
		self.rate100 = rospy.Rate(100)          # 100hz
		rospy.loginfo(rospy.get_caller_id() + "Test started...")



	def final_position(self,a,v,t,r):
		self.finalPosition=[4.65274715423584, -1.448388401662008, -1.9196799437152308, -1.3040936628924769, -4.758018557225363, -0.194709602986471]
		command = self.urSrciptToString(
			move="movej",
			jointPose=self.finalPosition,
			a=0.01,
			v=0.01,
			t=5,
			r=0)
		print command 
		self.MovePublisher.publish(command)
		time.sleep(10)

		# we need to check the orientation of the gripped object
		
		self.openGripper()
		print "Gripper open"


	def jointCallback(self,data):
		self.current_position= data.position
		# print self.goal_position
		pass

        # Keeping the node alive!
        #self.spin()


	def callback_tool(self, data):
		self.force_x = float(data.wrench.force.x)
		self.force_y = float(data.wrench.force.y)
		self.force_z = float(data.wrench.force.z)
		self.torque_x = float(data.wrench.torque.x)
		self.torque_y = float(data.wrench.torque.y)
		self.torque_z = float(data.wrench.torque.z)
		# print "Force in z= ", self.force_z

	def measure_initial_states(self):
		self.startingForce_x = self.force_x
		self.startingForce_y = self.force_y
		self.startingForce_z = self.force_z
		self.startingTorque_x = self.torque_x
		self.startingTorque_y = self.torque_y
		self.startingTorque_z = self.torque_z
		print "The initial force in z= ", self.startingForce_z
		print "The initial torque in y= ", self.startingTorque_y



if __name__ == '__main__':
	try:
		Picking()
	except rospy.ROSInterruptException:
		pass

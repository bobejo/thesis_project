#!/usr/bin/env python

import rospy, roslib
import time
import tf
from numpy import matrix
from numpy import linalg
import math
import numpy as np

# For inverse Kinematics
import tinyik

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from ur_msgs.msg import IOStates

from numpy.linalg import inv


# --------------------------------------------------------------------
class StereoBase():
    def __init__(self):
        rospy.init_node('StereoBase')

        self.listener = tf.TransformListener()

        # Subscribe to robot  (/calibrated_fts_wrench)
        rospy.Subscriber("/joint_states", JointState, self.cb1)

        # Publish to robot
        # self.urScriptPub = rospy.Publisher("/ur_driver/URScript", String, queue_size=1)

        self.MovePublisher = rospy.Publisher("/ur_driver/URScript", String, queue_size=1)

        # Go into spin, with rateOption!
        self.rate = rospy.Rate(10)  # 10hz
        rospy.loginfo(rospy.get_caller_id() + "Test started...")
        self.spin()

    # CBs ----------------------------------------------------------------
    # --------------------------------------------------------------------
    def cb1(self, data):
        # Get update from the manipulator:
        self.joints = data.position;
        # print self.joints;
        # print type(self.joints)
        # print "--------------------------"

    # spin --------------------------------------------------------------
    def spin(self):
        while (not rospy.is_shutdown()):
            start_time = rospy.get_rostime()
            # print time:
            # print("hello")
            end = time.time()
            # print(end - start)
            # print '%d------' % (self.n1)



            trans = self.FTSframe()

            # Go into spin, with rateOption!
            self.rate.sleep()

            # --------------------------------------------------------------------

    def FTSframe(self):
        try:

            self.listener.waitForTransform('/base', '/stereo_camera', rospy.Time(0), rospy.Duration(1))
            (trans, rot) = self.listener.lookupTransform('/base', '/stereo_camera', rospy.Time(0))

            # self.listener.waitForTransform('/stereo_camera','/base', rospy.Time(0),rospy.Duration(1))
            # (trans,rot) = self.listener.lookupTransform('/stereo_camera','/base', rospy.Time(0))


            transrotM = self.listener.fromTranslationRotation(trans, rot)  # Transformation matrix
            print
            transrotM

            rotationMat = transrotM[0:3, 0:3]  # Rotation
            translationMat = transrotM[:, 3]  # Translation
            # rotZ = np.matrix([
            # [math.cos(-3.14/2), -math.sin(-3.14/2), 0,0],
            # [math.sin(-3.14/2), math.cos(-3.14/2), 0,0],
            # [0, 0, 1,0],
            # [0,0,0,1],
            # ])

            # rotation_ee=rotationMat*rotZ  #Rotate end_effector frames to be like camera frame
            # R_b_ee=transrotM*rotZ
            # print rotation_ee
            # R_b_eee=inv(R_b_ee)

            # R_c_b = np.matrix([
            # [0, 1, 0,-900],
            # [-1, 0, 0,-240],
            # [0, 0, 1,1900],
            # [0,0,0,1],
            # ])


            point_t = (406, 356, -176, 1)  # triangulation

            point_t = np.matrix([406, 356, -176, 1])  # triangulation
            point_tt = point_t.transpose()
            R_c_ee = np.dot(transrotM, point_tt)
            # R_c_ee= transrotM*point_tt
            print
            R_c_ee

            # rot_matrix_y = matrix([[ math.cos(3.14), 0 ,math.sin(3.14)], [  0, 1, 0 ] , [ -math.sin(3.14), 0 ,math.cos(3.14)] ])
            # rotationMat= rot_matrix_y * rotationMat
            # # print rot_matrix_y
            # # print rot_matrix_y * rotationMat
            # # print self.RotationMatrix(R)


            # print transrotM

            # translationMat= transrotM[:,3]   # Translation
            # print transrotM
            # print translationMat

            # Transform a point from base frame to tool frame
            # trans_rot_mat=np.matrix(transrotM)
            # p_tool=trans_rot_mat*p_base   # The p point in tool frame
            # tool_transpose= p_tool.transpose()
            # to_list_tool=tool_transpose.tolist()  # Convert matrix into list
            # tool_pose_list=to_list_tool[0]

            return transrotM
            return R_c_ee

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print
            "EXCEPTION"
            # pass


# --------------------------------------------------------------------
# Here is the main entry point
if __name__ == '__main__':
    try:
        # Init the node:
        # rospy.init_node('Tutorial_UR10')
        # rospy.init_node('Test1')



        start = time.time()
        # Initializing and continue running the class Test1:
        StereoBase()

        # Just keep the node alive!
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
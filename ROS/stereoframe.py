#!/usr/bin/env python

import rospy
import time
import tf
from numpy import matrix
from numpy import linalg
import math

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


#--------------------------------------------------------------------
class CreateFrame():
    def __init__(self):
        
        # print(self.GraspingMatrix(matrix([[1],[2]])))
        
        # Start broadcaster:
        self.br = tf.TransformBroadcaster()
        # self.br2= tf.TransformBroadcaster()



        # self.listener = tf.TransformListener()




        # Subscribe to robot  (/calibrated_fts_wrench)
        rospy.Subscriber("/joint_states", JointState, self.cb1)
        
        # Publish to robot
        self.urScriptPub = rospy.Publisher("/ur_driver/URScript", String, queue_size=1)

        # Go into spin, with rateOption!
        self.rate = rospy.Rate(10)          # 10hz
        rospy.loginfo(rospy.get_caller_id() + "Test started...")
        self.spin()

#CBs ----------------------------------------------------------------
#--------------------------------------------------------------------
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

            trans=self.FTSframe()
            
            # Go into spin, with rateOption!
            self.rate.sleep()        

# --------------------------------------------------------------------
    def FTSframe(self):  
        try:      


            #Fixed Broadcaster
 

            self.br.sendTransform((-900, -240, 1910),
                            # (0.23172291, -0.6335338,  -0.06247095,  0.73555204), *90*math.pi/180
                            tf.transformations.quaternion_from_euler(math.pi, 0, 90*math.pi/180),
                            rospy.Time.now(),
                            "/stereo_camera", #child
                            "/base") #parent

            # self.br2.sendTransform((0, 0, 0),
            #                 # (0.23172291, -0.6335338,  -0.06247095,  0.73555204), *90*math.pi/180
            #                 tf.transformations.quaternion_from_euler(math.pi, 0, 0),
            #                 # tf.transformations.quaternion_from_euler(0, math.pi, 0),
            #                 rospy.Time.now(),
            #                 "/stereo_camera1", #child
            #                 "/stereo_camera") #parent

            #Dynamic Broadcster

            # t = rospy.Time.now().to_sec() * math.pi
            # self.br.sendTransform((2.0 * math.sin(t), 2.0 * math.cos(t), 0.0),
            #                 (0.0, 0.0, 0.0, 1.0),
            #                 rospy.Time.now(),
            #                 "/QRcode",
            #                 "/tool0_controller")


            print "-------------------------------------------------"
            # print R
            # return transrotM
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "EXCEPTION"
#             pass
            
#--------------------------------------------------------------------    
# Here is the main entry point
if __name__ == '__main__':
    try:
        # Init the node:
        rospy.init_node('Tutorial_UR')


        start = time.time()
        # Initializing and continue running the class Test1:
        CreateFrame()

        # Just keep the node alive!
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass

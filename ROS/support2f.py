#!/usr/bin/env python

class Support:
    def __init__(self):
        # Vars
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                            'wrist_2_joint', 'wrist_3_joint']

    # Methods-----------------------------------------------------------------------

    def degToRad(self, degPose):
        return map(lambda v: v / 180 * 3.14159, degPose)

    def urSrciptToString(self, move="movej", jointPose=[0, 0, 0, 0, 0, 0], a=1.0, v=0.2, t=8, r=0):
        return move + "(" + str(jointPose) + ", a=" + str(a) + ", v=" + str(v) + ", t=" + str(t) + ", r=" + str(r) + ")"

    def comparePose(self, pose):
        diffPose = [0] * 6
        for index in [0, 1, 2, 3, 4, 5]:
            diffPose[index] = list(self.joint_states[0].position)[index] - pose[index]
        return len(filter(lambda p: abs(p) > 0.5 / 180 * 3.14159, diffPose)) == 0

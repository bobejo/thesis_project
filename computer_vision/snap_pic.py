import os
import time as time
from time import gmtime


def snap_pic(name):
    """
    Takes two images one for each camera and saves them to the images/ folder

    :param name: Name of the file
    """
    t = get_time()
    os.system('wget http://admin:admin@192.168.1.138/dms?nowprofileid=1 -O' "images/left" + name + t + ".jpg")
    print("Picture 1 left is saved")
    os.system('wget http://admin:@192.168.1.144/dms?nowprofileid=1 -O' "images/right" + name + t + ".jpg")
    print("Picture 1 right is saved")


def get_time():
    t = time.strftime("%a, %d %b %Y %H_%M_%S", gmtime())
    t = t.replace(" ", "")
    return t[13:21]


# Choose name of pipe
snap_pic("pipe")

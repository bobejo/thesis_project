import os
import time as time
from time import gmtime

save_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\camera_calibration'


def snap_pic(name):
    """
    Takes two images one for each camera and saves them to the images/ folder.
    Requires program Wget.

    :param name: Save name of the file
    """
    t = get_time()
    os.system(
        'wget http://admin:admin@192.168.1.138/dms?nowprofileid=1 -O' '' + save_path + "\\left" + name + t + ".jpg")
    print("Picture 1 left is saved")
    os.system('wget http://admin:@192.168.1.144/dms?nowprofileid=1 -O' '' + save_path + "\\right" + name + t + ".jpg")
    print("Picture 1 right is saved")


def get_time():
    """
    Used for saving images with time in their name

    :return: The current time in format: hour:minutes:seconds
    """
    t = time.strftime("%a, %d %b %Y %H_%M_%S", gmtime())
    t = t.replace(" ", "")
    return t[13:21]

# Choose name of pipe
# snap_pic("Calibration")

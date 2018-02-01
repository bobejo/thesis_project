import os
import time as time
from time import gmtime


def main():
    snap_pic(1, "Big_pipe")


def snap_pic(nr_images, name):
    t = get_time()
    for x in xrange(1, nr_images + 1):
        os.system('wget http://admin:admin@192.168.1.138/dms?nowprofileid=1 -O' "images/left" + name + t + ".jpg")
        print("Picture 1 left is saved")
        os.system('wget http://admin:@192.168.1.144/dms?nowprofileid=1 -O' "images/right" + name + t + ".jpg")
        print("Picture 1 right is saved")


def get_time():
    t = time.strftime("%a, %d %b %Y %H:%M:%S", gmtime())
    t = t.replace(" ", "")
    return t[13:21]


main()

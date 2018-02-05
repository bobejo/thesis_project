#!/usr/bin/env python
'''
Based on https://github.com/opencv/opencv/blob/master/samples/python/mouse_and_match.py

mouse_and_match.py [-i path | --input path: default ../data/]
Demonstrate using a mouse to interact with an image:
 Read in the images in a directory one by one
 Allow the user to select parts of an image with a mouse
 When they let go of the mouse, it correlates (using matchTemplate) that patch with the image.
 SPACE for next image
 ESC to exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import argparse
import glob
# built-in modules
import os

import cv2 as cv

from snap_pic import get_time

drag_start = None
sel = (0, 0, 0, 0)
nr_clicked = 0
drag_end = None
save_path = '/home/saming/PycharmProjects/thesis_project/computer_vision/images/training_data/'


def onmouse(event, x, y, flags, param):
    global drag_start, sel, nr_clicked, drag_end
    if event == cv.EVENT_LBUTTONDOWN and nr_clicked < 2:
        if nr_clicked == 0:
            drag_start = x, y
            sel = 0, 0
            nr_clicked += 1
        else:
            drag_end = x, y
            nr_clicked += 1
    elif drag_start and drag_end:
        print(flags)
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.line(img, drag_start, drag_end, (0, 255, 255), 2)
            cv.imshow("Annotation", img)
            t = get_time()
            cv.imwrite(save_path + t + '_training.jpg', img)
        else:
            print("selection is complete")
            drag_start = None
            drag_end = None
            nr_clicked = 0


if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    parser.add_argument("-i", "--input",
                        default='/home/saming/PycharmProjects/thesis_project/computer_vision/images/cropped_images',
                        help="Input directory.")
    args = parser.parse_args()
    path = args.input

    cv.namedWindow("Annotation", 1)
    cv.setMouseCallback("Annotation", onmouse)
    '''Loop through all the images in the directory'''
    for infile in glob.glob(os.path.join(path, '*.*')):

        ext = os.path.splitext(infile)[1][1:]  # get the filename extension
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            img = cv.imread(infile, 1)
            if img is None:
                continue
            sel = (0, 0, 0, 0)
            drag_start = None
            cv.imshow("Annotation", img)
            if cv.waitKey() == 27:
                break

cv.destroyAllWindows()

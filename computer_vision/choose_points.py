#!/usr/bin/env python
"""
Based on https://github.com/opencv/opencv/blob/master/samples/python/mouse_and_match.py

Read an image and allow the user to select coordinates in the image which will be saved, draws a circle in the chosen position.

ESC to exit
"""

# Python 2/3 compatibility
from __future__ import print_function

import argparse
import glob
# built-in modules
import os

import cv2 as cv
import numpy as np

drag_start = None
sel = (0, 0, 0, 0)
nr_clicked = 0
drag_end = None

# image_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\camera_calibration\\left\\leftCalibration13_39_51.jpg'
# save='C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\left_move.npy'
save = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\right_move.npy'
image_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\camera_calibration\\right\\rightCalibration13_39_51.jpg'

drags = []
i = 1


def onmouse(event, x, y, flags, param):
    global drag_start, sel, nr_clicked, drag_end
    if event == cv.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0, 0


    elif drag_start:
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.circle(img, drag_start, 3, (0, 255, 255), 3)
            cv.imshow("Annotation", img)

        else:
            # print("selection is complete")
            cv.circle(img, drag_start, 3, (0, 255, 255), 3)
            drags.append(drag_start)
            drag_start = None


if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    parser.add_argument("-i", "--input",
                        default=image_path,
                        help="Input directory.")
    args = parser.parse_args()
    path = args.input

    cv.namedWindow("Annotation", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Annotation", onmouse)
    '''Loop through all the images in the directory'''
    print(path)
    allfiles = glob.glob(path)
    print(allfiles)
    for infile in allfiles:
        ext = os.path.splitext(infile)[1][1:]  # get the filename extension
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            img = cv.imread(infile, 1)
            if img is None:
                continue
            sel = (0, 0, 0, 0)
            drag_start = None
            cv.imshow('Annotation', img)
            if cv.waitKey() == 27:
                break
            np.save(save, drags)
            print(drags)

print(len(np.load(save)))
cv.destroyAllWindows()

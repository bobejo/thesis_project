#!/usr/bin/env python
'''
Based on https://github.com/opencv/opencv/blob/master/samples/python/mouse_and_match.py

 Read in the images in a directory one by one
 Allow the user to select two points and then draw a line between them.
 For each line drawn it save it as an jpg

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
import numpy as np

drag_start = None
sel = (0, 0, 0, 0)
nr_clicked = 0
drag_end = None
# Linux
save_path = '/home/saming/PycharmProjects/thesis_project/computer_vision/images/training_data/'
image_path = '/home/saming/PycharmProjects/thesis_project/computer_vision/images/cropped_images'
#Windows
#save_path='C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\images\\training_data'
#image_path='C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\images\\cropped_images'
move_path= '/home/saming/thesis_project/computer_vision/images/cropped_images/Annotated/'
drags = []
i=1

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
        if flags & cv.EVENT_FLAG_LBUTTON:
            cv.line(img, drag_start, drag_end, (0, 255, 255), 2)
            cv.imshow("Annotation", img)

        else:
           # print("selection is complete")
            drags.append([drag_start, drag_end])
            drag_start = None
            drag_end = None
            nr_clicked = 0


if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    parser.add_argument("-i", "--input",
                        default=image_path,
                        help="Input directory.")
    args = parser.parse_args()
    path = args.input

    cv.namedWindow("Annotation", 1)
    cv.setMouseCallback("Annotation", onmouse)
    '''Loop through all the images in the directory'''
    allfiles=glob.glob(path)
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

            blank_image = np.zeros((500, 350, 3), np.uint8)
            blank_image[:, :] = (100, 0, 0)
            for d in drags:
                cv.line(blank_image, d[0], d[1], (255, 255, 255), 10)
                #cv.circle(blank_image, (d[0]), 10, (255, 255, 255), -1)

            oldimg = infile.split('cropped_images/')
            print(i)
            i+=1
            cv.imwrite(save_path + oldimg[1], blank_image)
            drags = []
            drag_start = None
            drag_end = None
            nr_clicked = 0
            a = infile.split("cropped_images/")
            new_move_path = move_path + a[1]
            os.rename(infile, new_move_path)


cv.destroyAllWindows()

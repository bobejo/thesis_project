import cnn
import image_manipulation as im
import grasp_finder as gf
import snap_pic as sp
import paths
import keras
import glob
import global_variables as gv
import image_registration as ir
import numpy as np
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from Loss import LogLoss
import matplotlib.pyplot as plt
import global_variables as gv


def find_pipe(component):
    """
    Finds the best grasping point and angle of it.
    :param component: The type of pipe to pick,1 for pipe and 2 for u_pipe

    :return: The 3D gripping point in robot base frame and the gripping angle.
    """
    if component == 1:
        model = keras.models.load_model(paths.model_pipe_path)
    else:
        model = keras.models.load_model(paths.model_upipe_path)

    inp_gen = cnn.create_main_generator(paths.crop_path, 550, 400)
    limg = next(inp_gen)
    p = cnn.get_prediction(model, limg)
    p = cv2.resize(p[0], (420, 580))

    "Convert to binary image and dilate"

    bi = im.binary_image(p, gv.threshold)
    di = im.dilate_image(bi, gv.dilation_size)

    "Find the 2d gripping point and angle for the left image"
    cont, gripping_point_left, gripping_angle, _ = gf.contour_detector(di)

    gripping_point_left_full = (
        gripping_point_left[0] + gv.lcrop[component][1][0] - 1, gripping_point_left[1] + gv.lcrop[component][0][0] - 1)
    gripping_point_right_full = ir.affine_transformation(gv.A, gv.t, gripping_point_left_full)
    gripping_point_right_full = (int(round(gripping_point_right_full[0])), int(round(gripping_point_right_full[1])))

    return gripping_point_left_full, gripping_point_right_full, gripping_angle, cont, p


def find_oilfilter(limg):
    """
    Finds the best grasping point and angle of it.
    :param limg: The cropped image from left camera
    :return: The 3D gripping point in robot base frame and the gripping angle.
    """
    gripping_point_left, p = gf.circle_detector(limg)
    gripping_point_left_full = (
        gripping_point_left[0] + gv.lcrop[0][1][0] - 1, gripping_point_left[1] + gv.lcrop[0][0][0] - 1)
    gripping_point_right_full = ir.affine_transformation(gv.A_oilfilter, gv.t_oilfilter, gripping_point_left_full)
    gripping_point_right_full = (int(round(gripping_point_right_full[0])), int(round(gripping_point_right_full[1])))

    return gripping_point_left_full, gripping_point_right_full, 0, p


def find_3D_point(component):
    """
    Finds the best grasping point and angle of it.
    :param model: The loaded CNN model
    :param component: The type of object to pick, 0 for oilfilter ,1 for pipe and 2 for u_pipe
    :return: The 3D gripping point in robot base frame and the gripping angle.
    """

    "Take the images"
    l_img, r_img = sp.snap_pic('', paths.save_path)

    "Crop the image"
    im.crop_images((paths.save_path + '\\*jpg'), gv.lcrop[component], gv.rcrop[component])
    crop_path = paths.crop_path + '\\cropped\\left*'
    cropped_left_path = glob.glob(crop_path)  # The path of left cropped image
    cropped_left = cv2.imread(cropped_left_path[-1])

    "Decide what to pick"
    if component == 0:
        gripping_point_left_full, gripping_point_right_full, gripping_angle, cont = find_oilfilter(cropped_left)
    else:
        gripping_point_left_full,gripping_point_right_full, gripping_angle, cont, p = find_pipe(component)
        cv2.imshow('Prediction', p)

    # Find the points in right image

    # Triangulate
    lcm = np.genfromtxt(paths.left_matrix_path)
    rcm = np.genfromtxt(paths.right_matrix_path)

    tri_camera = gf.triangulate_point(gripping_point_left_full, gripping_point_right_full, rcm, lcm)

    # Convert to base frame
    gripping_point_base = ir.base_transform(gv.T, tri_camera)
    gripping_point_base = correct_gripping_point(gripping_point_base)

    # Plotting
    l_img = cv2.imread(l_img)
    r_img = cv2.imread(r_img)
    cv2.circle(l_img, gripping_point_left_full, 3, (255, 0, 0), 3)
    cv2.circle(r_img, gripping_point_right_full, 3, (255, 0, 0), 3)
    print('X: ', gripping_point_base[0])
    print('Y: ', gripping_point_base[1])
    print('Z: ', gripping_point_base[2])
    print('Gripping angle: ', gripping_angle, ' degrees')
    cv2.imshow('cont', cont)

    #cv2.imshow('fullLeft', l_img)
    #plt.imshow(r_img)
    #plt.show()
    cv2.waitKey(0)

    return [gripping_point_base[0][0], gripping_point_base[1][0], gripping_point_base[2][0]], gripping_angle


def correct_gripping_point(gripping_point_base):
    """
    Fixes the error of the reprojected grasping points.
    Not necessary if camera projection matrices correct
    :param The gripping point in base frame
    :return The corrected gripping point
    """
    z_error = 182
    min_y = -574.7
    max_y = -1031
    y_frac = min_y / max_y
    gripping_point_base[0] = gripping_point_base[0][0] - 190
    gripping_point_base[1] = gripping_point_base[1][0] - 55
    gripping_point_base[2] = gripping_point_base[2][0] + 100

    if gripping_point_base[1] / min_y > 1:
        gripping_point_base[2] = gripping_point_base[2] - (gripping_point_base[1] / max_y - y_frac) * z_error * 2

    return gripping_point_base


if __name__ == "__main__":

    for i in range(0, 100):
        gripping_points, gripping_angle = find_3D_point(0)
        #gripping_points, gripping_angle = find_3D_point(1)

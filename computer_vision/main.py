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


def find_3D_point():
    """
    Finds the best grasping point and angle of it.
    :return: The 3D gripping point in robot base frame and the gripping angle.
    """
    "Take the images and crop them"
    sp.snap_pic('', paths.save_path)
    im.crop_images(paths.save_path, gv.lcrop, gv.rcrop)
    crop_path = paths.save_path + '*croppedleft'
    cropped_left = glob.glob(crop_path)  # The left cropped image

    "Load trained model and run prediction"
    model = keras.models.load_model(paths.model_path)
    p = cnn.get_prediction(model, cropped_left[0])

    "Convert to binary image and dilate"
    p = im.image_segmentation(p[0], gv.threshold, gv.dilation_size)

    "Find the 2d gripping point and angle for the left image"
    _, gripping_point_left, gripping_angle = gf.contour_detector(p)

    "Find the corresponding point of the right camera. Requires the matrix A and t from "
    "image_registration.find_image_transformation()"

    gripping_point_right = ir.affine_transformation(gv.A, gv.t, gripping_point_left)

    "Finds the 3D point in camera frame, using triangulation, and convert to base frame."

    # Load camera matrices
    lcm = np.genfromtxt(paths.left_matrix_path)
    rcm = np.genfromtxt(paths.right_matrix_path)

    # Triangulate
    tri_camera = gf.triangulate_point(gripping_point_left, gripping_point_right, lcm, rcm)

    # Convert to base frame
    gripping_point_base = ir.camera_transform(gv.T, tri_camera)

    return gripping_point_base, gripping_angle


if __name__ == "__main__":
    find_3D_point()

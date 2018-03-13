import glob
import cv2
import numpy as np


def imgs2numpy(imgs_path, maximum_imgs):
    """
    Reads all the images of a path and returns an numpy array of the images. Will take a lot of memory therefore
    one can use the maximum_imgs to specify how many images you want.

    :param: The path to the images
    :param: The maximum amount of images wanted
    :return: The numpy array of size
    """

    image_list = sorted(glob.glob(imgs_path))
    if len(image_list) >= maximum_imgs:
        image_list = image_list[:maximum_imgs]
    i = 0
    np_img = 0
    for img in image_list:
        if i == 0:
            first_img = cv2.imread(img) / 255
            np_img = np.empty((len(image_list), first_img.shape[0], first_img.shape[1], first_img.shape[2]))
            np_img[i] = first_img
            i += 1
        else:
            np_img[i] = cv2.imread(img) / 255
            i += 1

    return np_img






import numpy as np
import glob
import cv2


def crop_images(path):
    """
    Takes multiple images and crops to each blue box.
    Cropping sizes are chosen manually

    :param path: The path to the folder where the images are
    """
    image_path_list = glob.glob(path)

    for image_path in image_path_list:
        img = cv2.imread(image_path, 1)
        a = image_path.split("images\\")

        if image_path.find('left') >= 0:
            cropped_image = img[400:900, 930:1280]  # Choose these rows,column
            image_path = a[0] + 'images\\' + 'cropped' + a[1]
            cv2.imwrite(image_path, cropped_image)

        elif image_path.find('right') >= 0:
            cropped_image = img[400:900, 1380:1730]  # Choose these rows,column
            image_path = a[0] + 'images\\' + 'cropped' + a[1]
            cv2.imwrite(image_path, cropped_image)
        else:
            print("File: " + a[1] + " skipped")


def binary_image(img, threshold):
    """
    Set all parameters below the threshold to 0 and all above to 1

    :param img: The image
    :param threshold: The threshold
    :return: An binary image where each pixel is either 1 or 0
    """
    r, bi = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return bi


def dilate_image(img, size):
    """
    Dilates the binary image, making the lines thicker and more connected

    :param img: Binary image
    :param size: Size of the dilation
    :return: Dilated binary image
    """
    kernel = np.ones((size, size))
    di = cv2.dilate(img, kernel, iterations=1)
    dilated_img = di.astype(np.uint8)
    return dilated_img


def image_segmentation(img, threshold, size):
    """
    Does image segmentation on the input image. First converts to binary image then dilates it

    :param img: The output from prediction
    :param threshold: The threshold for choosing 1 and 0
    :param size: The size of the dilation
    :return: A binary dilated image of type uint8 (wanted by blobdetector)
    """

    bi = binary_image(img[0], threshold)
    di = dilate_image(bi, size)
    dim = di.astype(np.uint8)
    return dim

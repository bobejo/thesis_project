import glob

import cv2


def crop_images(path):
    image_path_list = glob.glob(path)
    for image_path in image_path_list:
        img = cv2.imread(image_path, 1)
        cropped_image = img[100:200, 20:150]
        a, b = image_path.split("images/")
        image_path = a + "images/" + "cropped_images/" + b
        cv2.imwrite(image_path, cropped_image)


crop_images('/home/saming/PycharmProjects/thesis_project/computer_vision/images/*.jpg')

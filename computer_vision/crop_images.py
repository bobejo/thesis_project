import glob

import cv2


def crop_images(path):
    image_path_list = glob.glob(path)
    for image_path in image_path_list:
        img = cv2.imread(image_path, 1)
        if image_path.find('left') >= 0:
            cropped_image = img[500:900, 600:900]  # Choose these y,x
        else:
            cropped_image = img[400:1000, 1080:1350]
        a, b = image_path.split("images/")
        image_path = a + "images/" + "cropped_images/" + b
        cv2.imwrite(image_path, cropped_image)


#Choose where the camera files are
crop_images('/home/saming/PycharmProjects/thesis_project/computer_vision/images/*.jpg')

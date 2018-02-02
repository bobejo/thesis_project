import glob

import cv2


def crop_images(path):
    image_path_list = glob.glob(path)

    for image_path in image_path_list:
        img = cv2.imread(image_path, 1)
        a = image_path.split("images/")

        if image_path.find('left') >= 0:
            cropped_image = img[500:900, 600:900]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/" + a[1]
            cv2.imwrite(image_path, cropped_image)

        elif image_path.find('right') >= 0:
            cropped_image = img[500:900, 1050:1350]
            image_path = a[0] + "images/" + "cropped_images/" + a[1]
            cv2.imwrite(image_path, cropped_image)

        else:
            print("File: " + a[1] + " skipped")



#Choose where the camera files are
crop_images('/home/saming/PycharmProjects/thesis_project/computer_vision/images/*.jpg')

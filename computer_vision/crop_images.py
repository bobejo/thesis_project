import glob

import cv2


def crop_images(path):
    """
    Takes multiple images and crops to each blue box.
    Cropping sizes are chosen manually

    :param path: The path to the folder where the images are
    :return:
    """
    image_path_list = glob.glob(path)

    for image_path in image_path_list:
        img = cv2.imread(image_path, 1)
        a = image_path.split("images/")

        if image_path.find('left') >= 0:
            cropped_image = img[400:900, 260:610]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/oilpipe" + a[1]
            cv2.imwrite(image_path, cropped_image)

            cropped_image = img[400:900, 585:935]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/oilfilter" + a[1]
            cv2.imwrite(image_path, cropped_image)

            cropped_image = img[400:900, 950:1300]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/pipe" + a[1]
            cv2.imwrite(image_path, cropped_image)


        elif image_path.find('right') >= 0:
            cropped_image = img[400:900, 725:1075]  # Choose these rows,column
            image_path = a[0] + "images/" + "cropped_images/oilpipe" + a[1]
            cv2.imwrite(image_path, cropped_image)

            cropped_image = img[425:925, 1050:1400]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/oilfilter" + a[1]
            cv2.imwrite(image_path, cropped_image)

            cropped_image = img[400:900, 1400:1750]  # Choose these y,x
            image_path = a[0] + "images/" + "cropped_images/pipe" + a[1]
            cv2.imwrite(image_path, cropped_image)

        else:
            print("File: " + a[1] + " skipped")


# Choose where the camera files are
crop_images('/home/saming/PycharmProjects/thesis_project/computer_vision/images/*.jpg')

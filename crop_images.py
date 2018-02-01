import glob
import os

import cv2


def crop_images(path):
    image_list = glob.glob(path)
    # image_list2=os.listdir(path)
    print(image_list)
    for image in image_list:
        img = cv2.imread(image, 1)
        # print(img)
        cropped_image = img[100:200, 20:150]
        print('Size of the  image:' + str(img.shape))
        print('Size of the cropped image:' + str(cropped_image.shape))
        os.system('cd cropped_images')
        os.system('pwd')
        cv2.imwrite(image, cropped_image)


# crop_images("/home/saming/Documents/Master/images/*.jpg")
crop_images("/home/saming/Documents/Master/images/*.jpg")

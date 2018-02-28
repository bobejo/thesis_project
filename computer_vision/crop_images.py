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



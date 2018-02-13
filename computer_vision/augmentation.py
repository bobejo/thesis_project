import cv2
import glob
# The path where the images are loaded and saved
load_training = '/home/saming/thesis_project/computer_vision/images/training_data'
load_cropped = '/home/saming/thesis_project/computer_vision/images/cropped_images'



def rotate_images():
    """
    Loads the cropped and annotated images and rotates them 180 degrees.
    Saves the rotated images in a subfolder.
    """
    glob_crop=sorted(glob.glob(load_cropped+'/*.jpg'))
    glob_training = sorted(glob.glob(load_training + '/*.jpg'))
    for i in range(0,len(glob_crop)):

        img_crop=cv2.imread(glob_crop[i])
        img_training= cv2.imread(glob_training[i])
        img_crop=cv2.rotate(img_crop,cv2.ROTATE_180)
        img_training = cv2.rotate(img_training, cv2.ROTATE_180)
        crop_split=glob_crop[i].split("cropped_images/")
        train_split = glob_training[i].split("training_data/")

        save_crop=crop_split[0] + "cropped_images/Rotated/rotated_" + crop_split[1]
        save_training = train_split[0] + "training_data/Rotated/rotated_" + train_split[1]

        cv2.imwrite(save_crop,img_crop)
        cv2.imwrite(save_training, img_training)



rotate_images()
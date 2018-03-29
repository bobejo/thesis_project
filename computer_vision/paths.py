"""
Contains paths to the testing images and training images, the cnnmodel and camera matrices
Specify the path for the computer_vision folder
"""
# Path to computer_vision folder
project_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\'



## CNN
# Path where the training images are. Input images and target images
x_train_path = project_path+'\\images\\Training_data\\input_data'
y_train_path = project_path+'\\Training_data\\target_data'

# The testing images used for real time validation
x_test_path = project_path + '\\images\\Training_data\\test\\input'
y_test_path = project_path + '\\images\\Training_data\\test\\target'

# The verification images. Make sure that these are not used for training.
x_verification_path =project_path + '\\images\\Verification_data\\x_test\\test'
y_verification_path = project_path + '\\images\\Training_data\\y_test\\test'

# The path to the model
model_path = project_path+'\\models\\simple_model.h5'

## Triangulation
# The path to the camera matrices
left_matrix_path = project_path+'\\CameraMatrix\\leftMatrix.npy'
right_matrix_path =project_path+ '\\CameraMatrix\\rightMatrix.npy'

## Testing
# Used for affine transformation testing
test_path_right = project_path+'\\images\\Training_data\\test\\inp\\right\\right\\rightpipe08_41_37_2.jpg'
test_path_left = project_path+'\\images\\Training_data\\test\\inp\\left\\left\\leftpipe08_41_37_2.jpg'


# Affine transformation
# Path to left and right images with chessboard
# Gives points that will be used in the ransac leastsquare solver

left_chessboard =project_path+ 'images\\Chessboard_images\\leftcalibration07_16_21.jpg'
right_chessboard =project_path + 'images\\Chessboard_images\\rightcalibration07_16_21.jpg'
left_chessboard2 =project_path +'images\\Chessboard_images\\leftcalibration08_05_44.jpg'
right_chessboard2 =project_path +'images\\Chessboard_images\\rightcalibration08_05_44.jpg'


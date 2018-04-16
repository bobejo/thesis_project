"""
Contains paths to the testing images and training images, the cnnmodel and camera matrices
Specify the path for the computer_vision folder
"""
# Path to computer_vision folder
project_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\'

# Snap_pic save path
save_path = project_path+'\\images'
# Crop save path
crop_path = save_path+'\\cropped'


## CNN
# Path where the training images are. Input images and target images
x_train_path = project_path + '\\images\\Training_data\\input_data'
y_train_path = project_path + '\\images\\Training_data\\target_data'

# The testing images used for real time validation
x_validation_path = project_path + '\\images\\Validation_data\\input_data'
y_validation_path = project_path + '\\images\\Validation_data\\target_data'

# The verification images. Make sure that these are not used for training.
x_verification_path = project_path + '\\images\\Verification_data\\x_test\\test'
y_verification_path = project_path + '\\images\\Verification_data\\y_test\\test'

# The path to the model
model_path = project_path + '\\models\\upipe_long.h5'
model_pipe_path = project_path + '\\models\\upipe_long.h5'
model_upipe_path = project_path + '\\models\\upipe_long.h5'
simple_model_path = project_path + '\\models\\simple_model.h5'

## Triangulation
# The path to the camera matrices
left_matrix_path = project_path + '\\CameraMatrix\\lcm_vlh2.txt'
right_matrix_path = project_path + '\\CameraMatrix\\rcm_vlh2.txt'


"""
Contains paths to the testing images and training images, the cnnmodel and camera matrices

"""
## CNN
# Path where the training images are. Input images and target images
x_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\input_data'
y_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\target_data'

# The testing images used for real time validation
x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\input'
y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\target'

# The verification images. Make sure that these are not used for training.
x_verification_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Verification_data\\x_test\\test'
y_verification_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\y_test\\test'

# The path to the model
model_path = 'C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\models\\simple_model.h5'


## Triangulation
# The path to the camera matrices
left_matrix_path = 'CameraMatrix\\leftMatrix.npy'
right_matrix_path = 'CameraMatrix\\rightMatrix.npy'


## Testing
# Used for affine transformation testing
test_path_right = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\right\\right\\rightpipe08_41_37_2.jpg'
test_path_left = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\inp\\left\\left\\leftpipe08_41_37_2.jpg'

# Triangulation testing
test_path_right1 = 'C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\croppedleft_pipe08_18_23.jpg'
test_path_left1 = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\croppedright_pipe08_18_23.jpg'
test_path_right2 = 'C:\\Users\\Samuel\\Desktop\\pipes\\left\\images\\left_pipe08_18_23.jpg'
test_path_left2 = 'C:\\Users\\Samuel\\Desktop\\pipes\\right\\images\\right_pipe08_18_23.jpg'

"""
Contains paths to the testing images and training images, the cnnmodel and camera matrices

"""




# Path where the cropped images and training data is
# Windows
x_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\cropped_images'
y_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\target_data'
x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\input'
y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\test\\target'

# The path to the model
model_path='C:\\Users\\Samuel\\GoogleDrive\\Master\\Python\\thesis_project\\computer_vision\\models\\simple_model.h5'

# The path to the camera matrices
left_matrix_path='CameraMatrix\\leftMatrix.npy'
right_matrix_path='CameraMatrix\\rightMatrix.npy'

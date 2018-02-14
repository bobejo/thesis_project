from __future__ import print_function

import glob

import cv2
import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D
from keras.models import load_model
from matplotlib import pyplot as plt
# Tensorflow dimension ordering
K.set_image_dim_ordering('tf')

# Number of epochs the training should run
epochs = 1

# Path where the cropped images and training data is
# Windows
x_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\cropped_images\\*.jpg'
y_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\training_data\\*.jpg'
rotatex_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\cropped_images\\Rotated\\*.jpg'
rotatey_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\training_data\\Rotated\\*.jpg'
# Ubuntu
# x_train_path='/home/saming/thesis_project/computer_vision/images/cropped_images/*.jpg'
# y_train_path='/home/saming/thesis_project/computer_vision/images/training_data/*.jpg'
#rotatex_path='home/saming/thesis_project/computer_vision/images/cropped_images/Rotated/*.jpg'
#rotatey_path='/home/saming/thesis_project/computer_vision/images/training_data/Rotated/*.jpg'


# input image dimensions
img_rows, img_cols = 500, 350



def imgs2numpy():
    """
    Add the path to your chosen input and target images to the x_path and y_path for them to be added and saved as
    numpy arrays.
    """
    x_paths=[x_train_path, rotatex_path]
    y_paths = [y_train_path, rotatey_path]

    for j in range(0, len(x_paths)):

        x_path_list = glob.glob(x_paths[j])
        i = 0
        x_train = np.empty((len(x_path_list), 500, 350, 3))
        for ximg in x_path_list:
            x_train[i]= cv2.imread(ximg)/255
            i += 1
        savepath ='xtrain'+str(i)+'.npy'
        np.save(savepath, x_train)
        print('xtrain'+str(j+1)+'.npy'+ ' sucessfully saved')

        y_path_list = glob.glob(y_paths[j])
        i = 0
        y_train = np.empty((len(y_path_list), 500, 350, 3))
        for yimg in y_path_list:
            y_train[i] = cv2.imread(yimg)/255
            i += 1
        savepath = 'ytrain' + str(i) + '.npy'
        np.save(savepath, y_train)
        print('ytrain' + str(j+1) + '.npy' + ' sucessfully saved')


def train_model():
    """
    Loads the training and test data.
    Creates the model with the loss function and optimizer
    Trains the model with the training data.
    """

    x_train_cropped = np.load('xtrain.npy')
    y_train_cropped = np.load('ytrain.npy')
    x_train_rotate=np.load('xtrainrotate.npy')
    y_train_rotate=np.load('ytrainrotate.npy')
    x_train=np.vstack((x_train_cropped,x_train_rotate))
    y_train = np.vstack((y_train_cropped, y_train_rotate))
    print('Files loaded and stacked')

    # Will be another set of data later
    x_test=x_train_cropped
    y_test=y_train_cropped
    # x_test=np.load('xtest.npy')
    #y_test = np.load('ytest.npy')
    """
    # Make to float and normalise
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('float32')
    y_train /= 255
    print('Files converted and normalised')
    """
    inputs = keras.Input(shape=(img_rows, img_cols, 3))
    x1=Conv2D(30, (3, 3), padding='same', activation='relu')(inputs)
    x2=Conv2D(20, (3, 3), padding='same', activation='relu')(x1)
    D1 = Dropout(0.2)(x2)
    x3=Conv2D(20, (9, 9), padding='same', activation='relu')(D1)
    D2 = Dropout(0.4)(x3)
    x4=Conv2D(20, (1, 1), padding='same', activation='relu')(D2)
    x5=Conv2D(20, (1, 1), padding='same', activation='relu')(x4)
    x6_input= keras.layers.concatenate([x2, x5], axis=3)
    x6=Dropout(0.2)(x6_input)
    x7=Conv2D(3, (1, 1), padding='same', activation='relu')(x6)
    x8=Reshape((500,350,3))(x7)
    model=keras.Model(inputs,x8)

    model.summary()
    # initiate SGD optimizer and MSE loss function

    opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    # Start training
   # model.fit(x_train, y_train, epochs=epochs, batch_size=4, verbose=1, validation_data=(x_test, y_test), shuffle=True)


    # Save the model and all the weights
    model.save('dropout2.h5')
    print('Model saved')

def get_prediction():
    """
    Load the model with precomputed weights and does prediction with the loaded images

    :return: The output images from the keras prediction
    """
    model = load_model('1kbluemodel20.h5')
    x_train = np.load('xtrain.npy')
    predictions = model.predict(x_train[0:100], batch_size=2, verbose=1)

    return predictions


def compare_images():
    """
    Plots the input, target and prediction images.
    Runs get_prediction() to get the prediction.

    Press ANY button to change images.
    ESC to exit
    """
    p=get_prediction()
    x_train = np.load('xtrain.npy')
    y_train=np.load('ytrain.npy')
    for i in range(0,99):
        cv2.imshow('Input',  x_train[i])
        cv2.imshow('Target', y_train[i])
        cv2.imshow('Output', p[i])
        cv2.waitKey(0)



imgs2numpy()
#train_model()
#compare_images()

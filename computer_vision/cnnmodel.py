from __future__ import print_function
import keras
#from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D
from keras import backend as K

import cv2
import glob
import numpy as np

# Tensorflow dimension ordering
K.set_image_dim_ordering('tf')

epochs = 12
# Path where the cropped images and training data is
x_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\cropped_images\\*.jpg'
y_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\training_data\\*.jpg'

# input image dimensions
img_rows, img_cols = 500, 350


def img2numpy():
    #
    x_path_list = glob.glob(x_train_path)
    y_path_list = glob.glob(y_train_path)

    first=1
    for ximg in x_path_list:
        if first:
            x_train = cv2.imread(ximg)
            x_train = x_train.reshape([1, img_rows, img_cols, 3])
            first=0

        else:
            img=cv2.imread(ximg)
            img = img.reshape([1, img_rows, img_cols, 3])
            x_train = np.concatenate((x_train, img), 0)


    first = 1
    for yimg in y_path_list:
        if first:
            y_train = cv2.imread(yimg)
            y_train = y_train.reshape([1, img_rows, img_cols, 3])
            first = 0

        else:
            img = cv2.imread(yimg)
            img = img.reshape([1, img_rows, img_cols, 3])
            y_train = np.concatenate((y_train, img), 0)

    np.save('xtrain.npy',x_train)
    np.save('ytrain.npy',y_train)


def train_model():
    x_train=np.load('xtrain.npy')
    y_train=np.load('ytrain.npy')

    # Will be another set of data later
    x_test=x_train
    y_test=y_train

    # Some versions take the channel first and some last
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        y_train = y_train.reshape(y_train.shape[0], 3, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 1)

    # Make to float and normalise
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = y_train.astype('float32')
    y_train /= 255

    inputs = keras.Input(shape=(500,350,3))
    x1=Conv2D(30, (3, 3), padding='same', activation='relu')(inputs)
    x2=Conv2D(20, (3, 3), padding='same', activation='relu')(x1)
    x3=Conv2D(20, (9, 9), padding='same', activation='relu')(x2)
    x4=Conv2D(20, (1, 1), padding='same', activation='relu')(x3)
    x5=Conv2D(20, (1, 1), padding='same', activation='relu')(x4)
    x6_input= keras.layers.concatenate([x2, x5], axis=3)
    x6=Conv2D(3, (1, 1), padding='same', activation='relu')(x6_input)
    model=keras.Model(inputs,x6)

    #model.summary()

    # initiate SGD optimizer and MSE loss function

    opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    # Start training
    model.fit(x_train, y_train, epochs=epochs, batch_size=10, verbose=1, validation_data=(x_test, y_test), shuffle=True)


    # Save the model and all the weights
    model.save('funcmodel.h5')


from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import glob
import cv2
import keras
import numpy as np
import img_numpy
from Loss import logloss, accuracy
from vis_activations import display_activations, get_activations
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import paths
from random import randint
from scipy import ndimage
import random

# Tensorflow dimension ordering
K.set_image_dim_ordering('tf')

# input image dimensions
img_rows, img_cols = 550, 400


def create_main_generator(input_path, row_size, col_size):
    data_gen_args = dict(rescale=1. / 255)

    input_datagen = ImageDataGenerator(**data_gen_args)
    input_generator = input_datagen.flow_from_directory(
        input_path,
        class_mode=None,
        batch_size=1,
        target_size=(row_size, col_size))
    return input_generator


def create_generators(input_path, target_path, batch_size, row_size, col_size):
    """
    Creates a two generator with the training data (input images and target images) and returns a zipped generator.
    The generator first rescales the input and target images and then randomly flips and shifts them in the same way.

    :param input_path:  The path of the input images that will be used.
    :param target_path: The path of the target images that wil be used.
    :param batch_size: The number of images generated per batch.
    :param row_size: The wanted row size of the generated image
    :param col_size: The wanted column size of the generated image
    :return: Tuple with the two zipped generators
    """

    # The arguments
    data_gen_args = dict(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.03,
                         height_shift_range=0.03, channel_shift_range=0.)

    input_datagen = ImageDataGenerator(**data_gen_args)
    target_datagen = ImageDataGenerator(**data_gen_args)

    # Use the same seed
    seed = randint(1, 1000)

    input_generator = input_datagen.flow_from_directory(
        input_path,
        class_mode=None,
        batch_size=batch_size,
        target_size=(row_size, col_size),
        seed=seed)

    target_generator = target_datagen.flow_from_directory(
        target_path,
        class_mode=None,
        batch_size=batch_size,
        target_size=(row_size, col_size),
        color_mode='grayscale',
        seed=seed)

    # Combine generators into one which yields input images and target images
    train_generator = zip(input_generator, target_generator)

    return train_generator


def train_model(x_train_path, y_train_path, x_test_path, y_test_path, batch_size, steps_per_epoch, training_epochs):
    """
    Creates the model with the loss function and optimizer
    Creates the generators for generating input and target images
    Trains the model with the training data and saves it when done.

    :param x_train_path: The path of the input training images
    :param y_train_path: The path of the target training images
    :param x_test_path: The path of the input validation images
    :param y_test_path: The path of the target validation images
    :param batch_size: The number of images to be generated per batch
    :param steps_per_epoch: The number of images used per epoch
    :param epochs: The number of epochs the training is done for
    """

    model = create_simple_model()
    train_generator = create_generators(x_train_path, y_train_path, batch_size, img_rows, img_cols)
    validation_generator = create_generators(x_test_path, y_test_path, batch_size, img_rows, img_cols)
    saveclk = keras.callbacks.ModelCheckpoint('models\\clbkmodel.h5', monitor='val_loss', verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False, mode='min', period=1)
    clbk = [saveclk]
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=training_epochs,
                        validation_data=validation_generator, callbacks=clbk,
                        validation_steps=50)  # Number of images used for validation

    # model.save('models\\simp_model.h5')
    print('Model saved')


def create_simple_model():
    """
    Creates an simple model with optimizer and loss function
    Memory friendly
    :return: A simple CNN model
    """
    inputs = keras.Input(shape=(550, 400, 3))
    x1 = BatchNormalization()(inputs)
    x1 = Conv2D(30, (3, 3), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x1)
    x2 = Conv2D(20, (3, 3), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x2)
    x3 = Conv2D(20, (9, 9), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x3)
    x4 = Conv2D(20, (1, 1), padding='same', activation='relu')(x4)
    x5 = BatchNormalization()(x4)
    x5 = Conv2D(20, (1, 1), padding='same', activation='relu')(x5)
    x6 = BatchNormalization()(x5)
    x6_input = keras.layers.concatenate([x2, x6], axis=3)
    x7 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x6_input)
    model = keras.Model(inputs, x7)
    opt = keras.optimizers.SGD()
    model.compile(loss='mse', optimizer=opt)

    return model


def create_model():
    inputs = keras.Input(shape=(550, 400, 3))
    x1 = BatchNormalization()(inputs)
    x1 = Conv2D(30, (3, 3), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x1)
    x2 = Conv2D(20, (3, 3), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x2)
    x3 = Conv2D(20, (9, 9), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x3)
    x4 = Conv2D(20, (1, 1), padding='same', activation='relu')(x4)
    x5 = BatchNormalization()(x4)
    x5 = Conv2D(20, (1, 1), padding='same', activation='relu')(x5)
    x6 = BatchNormalization()(x5)
    x6_input = keras.layers.concatenate([x2, x6], axis=3)
    x7 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x6_input)
    model = keras.Model(inputs, x7)
    opt = keras.optimizers.SGD()
    model.compile(loss='mse', optimizer=opt)

    return model


def get_prediction(model, x_test):
    """
    Does keras.prediction with the testing data.

    :param model: The model the prediction will be done on
    :param x_test: The testing images as numpy arrays
    :return: p: The prediction output
    """
    if len(x_test.shape) < 4:
        x_test.reshape(1, x_test.shape[0], x_test.shape[1], x_test.shape[2])
    p = model.predict(x_test, batch_size=2, verbose=1)
    return p

# train_model(paths.x_train_path, paths.y_train_path, paths.x_validation_path, paths.y_validation_path, 2, 10, 5)


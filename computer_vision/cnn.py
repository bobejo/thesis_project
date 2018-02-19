from __future__ import print_function
import glob
import cv2
import keras
import numpy as np
from Loss import logloss
from vis_activations import display_activations, get_activations
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Tensorflow dimension ordering
K.set_image_dim_ordering('tf')

# Number of epochs the training should run
epochs = 1

# Path where the cropped images and training data is
# Windows
x_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\cropped_images'
y_train_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Training_data\\target_data'
x_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Verification\\x_test'
y_test_path = 'C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\Verification\\y_test'
# Ubuntu
# x_train_path='/home/saming/thesis_project/computer_vision/images/cropped_images/*.jpg'
# y_train_path='/home/saming/thesis_project/computer_vision/images/training_data/*.jpg'
# rotatex_path='home/saming/thesis_project/computer_vision/images/cropped_images/Rotated/*.jpg'
# rotatey_path='/home/saming/thesis_project/computer_vision/images/training_data/Rotated/*.jpg'


# input image dimensions
img_rows, img_cols = 500, 350


def imgs2numpy():
    """
    Add the path to your chosen input and target images to the x_path and y_path for them to be added and saved as
    numpy arrays. Saves them in 4 files that can later be stack. This is because of memory issues
    """

    # Test sets
    x_path_list = sorted(glob.glob(x_train_path + 'input\\*.jpg'))
    y_path_list = sorted(glob.glob(y_train_path + 'input\\*.jpg'))
    x_path_list = x_path_list[:100]
    y_path_list = y_path_list[:100]
    x_test = np.empty((len(x_path_list), 500, 350, 3))
    y_test = np.empty((len(y_path_list), 500, 350, 3))
    i = 0
    for ximg in x_path_list:
        x_test[i] = cv2.imread(ximg) / 255
        i += 1
    np.save('xtest.npy', x_test)
    i = 0
    for yimg in y_path_list:
        y_test[i] = cv2.imread(yimg) / 255
        i += 1
    np.save('ytest.npy', y_test)
    print('TestImages saved')
    for j in range(0, 1):

        for k in range(0, 4):
            """
            x_path_list = sorted(glob.glob(x_train_path))
            i = 0
            if k==0:
                x_path_list=x_path_list[:len(x_path_list)//4]
            elif k==1:
                x_path_list = x_path_list[len(x_path_list) // 4:len(x_path_list) // 2]
            elif k==2:
                x_path_list = x_path_list[len(x_path_list)//2:len(x_path_list) // 2 +len(x_path_list) // 4]
            else:
                x_path_list = x_path_list[len(x_path_list) // 2 + len(x_path_list) // 4:]

            x_train = np.empty((len(x_path_list), 500, 350, 3))
            for ximg in x_path_list:
                x_train[i] = cv2.imread(ximg) / 255
                i += 1

            savepath = 'xtrain' + str(j)+ str(k) + '.npy'
            np.save(savepath, x_train)
            print('xtrain' + str(j)+ str(k) + '.npy' + ' sucessfully saved')

            
            y_path_list = sorted(glob.glob(y_paths[j]))
            print(len(y_path_list))
            if k==0:
                y_path_list=y_path_list[:len(y_path_list)//4]
            elif k==1:
                y_path_list = y_path_list[len(y_path_list) // 4:len(y_path_list) // 2]
            elif k==2:
                y_path_list = y_path_list[len(y_path_list)//2:len(y_path_list) // 2 +len(y_path_list) // 4]
            else:
                y_path_list = y_path_list[len(y_path_list) // 2 + len(y_path_list) // 4:]

            i = 0
            print(len(y_path_list))
            y_train = np.empty((len(y_path_list), 500, 350, 3))
            print(len(y_train))
            for yimg in y_path_list:
                y_train[i] = cv2.imread(yimg)/255
                i += 1
            savepath = 'ytrain' + str(j)+ str(k) + '.npy'
            np.save(savepath, y_train)
            print('ytrain' + str(j)+ str(k) + '.npy' + ' sucessfully saved')
            """


def try_generator(generator):
    """
    Grabs an output from the generator and plots it.

    :param generator: The generator you want to try.
    """
    for i in range(0, 100):
        x, y = next(generator)
        cv2.imshow('x', x[0])
        cv2.imshow('y', y[0])
        cv2.waitKey(0)


def create_generators(input_path, target_path, batch_size):
    """
    Creates a two generator with the training data (input images and target images) and returns a zipped generator.

    :param input_path:  The path of the input images that will be used.
    :param target_path: The path of the target images that wil be used.
    :param batch_size: The number of images generated per batch.
    :return: Tuple with the two zipped generators
    """

    # The arguments
    data_gen_args = dict(rescale=1. / 255)

    input_datagen = ImageDataGenerator(**data_gen_args)
    target_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed
    seed = 1

    input_generator = input_datagen.flow_from_directory(
        input_path,
        class_mode=None,
        batch_size=batch_size,
        target_size=(img_rows, img_cols),
        seed=seed)

    target_generator = target_datagen.flow_from_directory(
        target_path,
        class_mode=None,
        batch_size=batch_size,
        target_size=(img_rows, img_cols),
        color_mode='grayscale',
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(input_generator, target_generator)

    return train_generator


def train_model(batch_size, steps_per_epoch, epochs):
    """
    Creates the model with the loss function and optimizer
    Creates the generators for generating input and target images
    Trains the model with the training data and saves it when done

    :param batch_size: The number of images to be generated per batch
    :param steps_per_epoch: The number of images used per epoch
    :param epochs: The number of epochs the training is done for
    """

    model = create_simple_model()
    model.summary()

    train_generator = create_generators(x_train_path, y_train_path, batch_size)

    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save the model and all the weights
    model.save('loopsig.h5')
    print('Model saved')


def create_simple_model():
    """
    Creates an simple model with optimizer and loss function

    :return: A simple CNN model
    """
    inputs = keras.Input(shape=(img_rows, img_cols, 3))
    x1 = Conv2D(30, (3, 3), padding='same', activation='relu')(inputs)
    x2 = Conv2D(20, (3, 3), padding='same', activation='relu')(x1)
    x3 = Conv2D(20, (9, 9), padding='same', activation='relu')(x2)
    x4 = Conv2D(20, (1, 1), padding='same', activation='relu')(x3)
    x5 = Conv2D(20, (1, 1), padding='same', activation='relu')(x4)
    x6_input = keras.layers.concatenate([x2, x5], axis=3)
    x7 = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(x6_input)
    model = keras.Model(inputs, x7)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model


def compare_images(model, x_path, y_path):
    """
    Plots the input, target and prediction images.
    Runs get_prediction() to get the prediction.

    Press ANY button to change images.
    ESC to exit

    :param model: The model the prediction will be done on
    :param x_path: The path to the testing input images
    :param y_path: The path to the target images
    """

    x_test = np.load(x_path)
    y_test = np.load(y_path)
    p = model.predict(x_test[0:100], batch_size=2, verbose=1)
    print(type(p))
    print(p.shape)
    for i in range(0, 100):
        ga = get_activations(model, x_test[i])
        display_activations(ga)
        cv2.imshow('input', x_test[i])
        cv2.imshow('target', y_test[i])
        cv2.imshow('output', p[i])
        cv2.waitKey(0)


# imgs2numpy()
# train_model()
# compare_images()


train = create_generators(x_train_path, y_train_path)
try_generator(train)

from __future__ import print_function
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from keras.datasets import cifar10
import cv2
import glob
import numpy as np

K.set_image_dim_ordering('tf')

epochs = 12

x_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\cropped_images\\*.jpg'
y_train_path='C:\\Users\\Samuel\\GoogleDrive\Master\Python\\thesis_project\\computer_vision\\images\\training_data\\*.jpg'
# input image dimensions
img_rows, img_cols = 500, 350
# the data, shuffled and split between train and test sets
#x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
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
'''

x_train=np.load('xtrain.npy')
y_train=np.load('ytrain.npy')


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    y_train = y_train.reshape(y_train.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255
y_train = y_train.astype('float32')
y_train /= 255


model = Sequential()
model.add(Conv2D(30, (3, 3), padding='same',input_shape=(500,350,3)))
model.add(Activation('relu'))

model.add(Conv2D(20, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(20, (9, 9), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(20, (1, 1), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(20, (1, 1), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(3, (1, 1), padding='same'))
model.add(Activation('sigmoid'))

#model.add(Reshape((500,350,3)))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()
#model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_data=x_test, shuffle=True)

#model.save('my_model.h5')

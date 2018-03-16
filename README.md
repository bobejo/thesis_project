# Dynamic bin picking of complex details and placing

This code contains the computer vision and ROS used for the project

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites


* [Numpy, Scipy and Matplotlib](https://www.scipy.org/install.html "Scipy installation")
* [Opencv](https://pypi.python.org/pypi/opencv-python "Opencv installation")
* [Tensorflow](https://www.tensorflow.org/install/ "Tensorflow installation")
* [Keras](https://keras.io/#installation "Keras installation")
* [h5py](http://docs.h5py.org/en/latest/build.html "h5 installation")
* [ROS Indigo](http://wiki.ros.org/indigo/Installation "ROS installation")

### Installing

Download the project using:

```
git clone https://github.com/bobejo/thesis_project.git
```
Make sure to change to path of your images, models and camera matrices in the file **computer_vision/paths.py**

## Computer vision

The position of the grasping point is found in the following way:

1. Each camera takes an image.
2. One of the image is cropped and propagated through the neural network.
3. The output image is converted to a binary image using thresholding and then dilated.
4. A contour detector is used to find the center of the largest area and its angle.
5. The corresponding point is found in the second image.
6. Triangulation is used to find the 3D coordinate.
7. The coordinate is transformed to base frame and sent, together with the angle.
## ROS

## Authors

* **Mahmoud Hanafy** - hanafy@student.chalmers.se
* **Samuel Ingemarsson** - samuel.ingemarsson@gmail.com	


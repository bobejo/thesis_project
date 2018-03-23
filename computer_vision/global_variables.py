import numpy as np

# Cropping
# VLH
lcrop = [(200, 750), (180, 550)]
rcrop = [(125, 675), (530, 900)]

# Binary threshold
threshold = 0.2
# Dilation size
dilation_size = 3

# Rotation matrix and translation
# Use image_registration.find_image_transformation()
A = np.array([[1.01757231, -0.13117867], [0.14959926, 1.00347523]])
t = np.array([307.74977789, -262.3793941])

# Transformation matrix between camera frame and base frame
T = np.array([[0, 1, 0, -900], [1, 0, 0, -240], [0, 0, 1, 1910]])

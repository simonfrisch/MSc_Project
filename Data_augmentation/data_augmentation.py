import os
import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import rotate
from skimage.util import random_noise

# inspiration: https://towardsdatascience.com/image-augmentation-using-python-numpy-opencv-and-skimage-ef027e9898da

os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\Archiv\\images")
files = os.listdir()

# flip image (rotate 180 degrees)
for file in files:
    img = cv.imread(file)
    img_flipped = np.flipud(img)
    img_name = file + "_flip.jpg"
    cv.imwrite(img_name, img_flipped)

# rotate image to some degrees
for file in files:
    img = cv.imread(file)
    img_rotate = rotate(img, angle=45)
    img_name = file + "_rotate.jpg"
    imsave(img_name, img_rotate)

# add noise to the image
for file in files:
    img = cv.imread(file)
    img_noise = random_noise(img, seed=123)
    img_name = file + "_noise.jpg"
    imsave(img_name, img_noise)

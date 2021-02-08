import os
import imageio
import numpy as np
from skimage.transform import rotate
from skimage.io import imsave, imread

path = "D:\\Documents\\Orig"

def rotate_image(path): # rotate image by 45 degrees
    os.chdir(path)
    files = os.listdir()
    for file in files:
        img = imageio.imread(file)
        img_rotate = rotate(img, angle=45, cval=1)
        img_rotate = (img_rotate*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_rotate" + file[-4:]
        imsave(img_name, img_rotate)

def zoom_image(path):
    os.chdir(path)
    files = os.listdir()
    for f in files:
        img = imread(f)
        img = img[1050:2500, 1000:2900, :] # tugraz: img[900:3100, 1300:4700, :] # img = img[90:360, 130:530, :]
        img_name = "zoom_" + f
        imsave(img_name, img)

zoom_image(path)

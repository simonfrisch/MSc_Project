import os
import imageio
import numpy as np
from skimage.io import imsave, imread
from skimage.util import random_noise
from skimage.filters import gaussian, unsharp_mask
from skimage.transform import rotate

path = "D:\\Documents\\Original"

def add_salt_noise(path): # adding salt noise
    os.chdir(path)
    files = os.listdir()
    for file in files:
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='salt', seed=123) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_salt" + file[-4:]
        imsave(img_name, img_noise)

def add_gaussian_noise(path): # adding salt noise
    os.chdir(path)
    files = os.listdir()
    for file in files:
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='gaussian', seed=123) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_gaussian" + file[-4:]
        imsave(img_name, img_noise)

def gaussian_filter(path): # gaussian smoothing/blurring
    os.chdir(path)
    files = os.listdir()
    for file in files:
        img = imread(file) # uint8
        img_filt = gaussian(img) # float64
        img_filt = (img_filt * 255).astype(np.uint8)  # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_gaussian_filter" + file[-4:]
        imsave(img_name, img_filt)

def unsharp(path): # sharpen image (filter is called "unsharp_mask"
    os.chdir(path)
    files = os.listdir()
    for file in files:
        img = imread(file) # uint8
        img_filt = unsharp_mask(img) # float64
        img_filt = (img_filt * 255).astype(np.uint8)  # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_unsharped" + file[-4:]
        imsave(img_name, img_filt)

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

unsharp(path)

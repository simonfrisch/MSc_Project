import os
import cv2
import numpy as np
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.filters import gaussian, median, unsharp_mask
from skimage.transform import rotate

path = "C:\\Users\\Simon\\Downloads\\Aargau_25_high"
path2 = "C:\\Users\\Simon\\Downloads\\Aargau_25_high_augm"

def resize_img(path):
    os.chdir(path)
    files = os.listdir()
    counter = 1
    for file in files:
        img = imread(file) # uint8
        img_resize = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST) # high: 640x480, low: 480x320 # INTER_LINEAR INTER_NEAREST
        #img_name = "UZH_" + str(counter) + "_original.JPG"
        end = file.find(".")
        img_name = file[:end] + "_high" + file[-4:]
        imsave(img_name, img_resize)
        counter += 1

def rename_img(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        end = file.find(".")
        img_name = file[:end] + "_remove_row" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img)

def add_gaussian_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='gaussian', seed=123) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_gaussian" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_noise)

def add_poisson_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='poisson', seed=123) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_poisson" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_noise)

def add_salt_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='salt', seed=1234) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_salt" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_noise)

def add_pepper_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='pepper', seed=42) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_pepper" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_noise)

def add_salt_and_pepper_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_noise = random_noise(img, mode='s&p', seed=0) # float64
        img_noise = (img_noise*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_noise_salt_pepper" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_noise)

def gaussian_filter(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm =  gaussian(img, sigma=1) # float64
        img_augm = (img_augm*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_gaussian_filter" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def median_filter(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm =  median(img, mode='nearest')
        end = file.find(".")
        img_name = file[:end] + "_median_filter" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

'''def hysteresis_threshold(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = apply_hysteresis_threshold(img, 140, 143)
        img_augm = (img_augm*255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_hysteresis_threshold" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)'''

def rotate_image(path):
    os.chdir(path)
    files = os.listdir()
    counter = 1
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_resize = rotate(img, 180)
        end = file.find(".")
        img_name = file[:end] + "_rotate" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_resize)
        counter += 1

def change_contrast_brightness1(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = cv2.convertScaleAbs(img, alpha=1.6, beta=0) # alpha controls contrast (alpha=1 means no change)
        end = file.find(".")
        img_name = file[:end] + "_contrast_brightness_1" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def change_contrast_brightness2(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = cv2.convertScaleAbs(img, alpha=1, beta=50) # beta controls brightness (beta=0 means no change)
        end = file.find(".")
        img_name = file[:end] + "_contrast_brightness_2" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def change_contrast_brightness3(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        end = file.find(".")
        img_name = file[:end] + "_contrast_brightness_3" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def change_contrast_brightness4(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
        end = file.find(".")
        img_name = file[:end] + "_contrast_brightness_4" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def graysacle(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        end = file.find(".")
        img_name = file[:end] + "_grayscle" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def flip_image(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = img[::-1]
        end = file.find(".")
        img_name = file[:end] + "_flipped" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def sharpen_image(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = unsharp_mask(img)
        img_augm = (img_augm * 255).astype(np.uint8)  # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_sharpened" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def crop_image(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        a = int(img.shape[0]*0.6) # use only 60% of the pixels
        b = int(img.shape[1]*0.6) # use only 60% of the pixels
        img_augm = img[:a,:b,:]
        img_augm = cv2.resize(img_augm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        end = file.find(".")
        img_name = file[:end] + "_cropped" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def dropout_image(path):
    os.chdir(path)
    files = os.listdir()
    np.random.seed(1000) # this makes sure to use the same seed for both the images and the mask dropout
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = img
        begin = np.random.randint(20, 300)
        dist = 25
        img_augm[begin:begin+dist,begin:begin+dist,:] = 0
        begin2 = np.random.randint(250, 450)
        dist = 30
        img_augm[begin2-begin:begin2+dist,begin2:begin2+dist,:] = 0
        end = file.find(".")
        img_name = file[:end] + "_dropout" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def gaussian_filter_noise(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = gaussian(img, sigma=1.5) # float64
        img_augm = random_noise(img_augm, mode='salt', seed=111)
        img_augm = (img_augm * 255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_gaussian_filter_noise" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

def sharpen_noise_image(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = unsharp_mask(img)
        img_augm = random_noise(img_augm, mode='pepper', seed=50)
        img_augm = (img_augm * 255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_sharpened_noise" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

'''def remove_row(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = img[::2,::2,:] # remove every other row
        img_augm = unsharp_mask(img_augm) # unsharp image
        img_augm = cv2.resize(img_augm, (480, 320)) # high: 640x480, low: 480x320
        img_augm = (img_augm * 255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_remove_row" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)'''

def remove_row(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        os.chdir(path)
        img = imread(file) # uint8
        img_augm = gaussian(img, sigma=1.5)  # float64
        img_augm = random_noise(img_augm, mode='salt', seed=100)
        img_augm = (img_augm * 255).astype(np.uint8) # convert from float64 to unit8
        end = file.find(".")
        img_name = file[:end] + "_remove_row" + file[-4:]
        os.chdir(path2)
        imsave(img_name, img_augm)

#add_gaussian_noise(path)
#add_poisson_noise(path)
#add_salt_noise(path)
#add_pepper_noise(path)
#add_salt_and_pepper_noise(path)
#gaussian_filter(path)
#median_filter(path)
#rotate_image(path)  # masks augmenting
#change_contrast_brightness1(path)
#change_contrast_brightness2(path)
#change_contrast_brightness3(path)
#change_contrast_brightness4(path)
#graysacle(path)
#flip_image(path)  # masks augmenting
#sharpen_image(path)
#crop_image(path)  # masks augmenting
#dropout_image(path)  # masks augmenting
#gaussian_filter_noise(path)
#sharpen_noise_image(path)
#remove_row(path)

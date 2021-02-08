import os
import cv2 as cv
import numpy as np

os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\semantic_drone_dataset\\training_set\\gt\\semantic\\label_images_downsampled") # segmentation
files = os.listdir()

for file in files:
    img = cv.imread(file)
    img[np.where((img != [168,42,28]).all(axis=2))] = [255,255,255] # set pixels with other BGR code to white
    img[np.where((img == [168,42,28]).all(axis=2))] = [0,0,0] # set all blue pixels to black
    pic_name = file + '_' + 'binary_mask.png'
    cv.imwrite(pic_name, img)

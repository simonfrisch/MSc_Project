import numpy as np
import os
import cv2 as cv

os.chdir("D:\\Documents\\UZH\FS20\\MSc Project\\Python\\OpenCV\Misc")
imgloc = "Seerobbe.jpg"
img = cv.imread(imgloc)

# Blur filter
kernel = np.ones((3,3),np.float32)/9
img_blur = cv.filter2D(img,-1,kernel)
cv.imshow('image_see',img_blur)
cv.waitKey(0)

# Sharpening
kernel1 = np.ones((3,3),np.float32)/9
kernel2 = np.zeros((3,3), np.float32)
kernel2[1,1] = 2
kernel = kernel2 - kernel1
img_blur = cv.filter2D(img,-1,kernel)
cv.imshow('image_see',img_blur)
cv.waitKey(0)


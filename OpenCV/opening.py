import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("D:\\Documents")
img = cv2.imread('img_0_mask.png', 0) # cv2.IMREAD_UNCHANGED instead of 0 to avoid gray conversion -> this maybe needs other SE
kernel = np.ones((3,3), np.uint8) # Structural Element (SE)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#plt.imshow(img) # show original image
#plt.show()
#plt.imshow(opening) # show opening image
#plt.show()

cv2.imwrite("img_opening.png", opening)

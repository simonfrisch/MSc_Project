import os, cv2
from skimage.io import imread, imsave
path = 'C:\\Users\\Simon\\Desktop'
os.chdir(path)
f = "UZH_9_mask.png"
img = imread(f)
img_resize = cv2.resize(img, (480, 320), interpolation=cv2.INTER_AREA)
imsave("UZH_9_low_inter_area.png", img_resize)



import os, cv2
from skimage.io import imread, imsave
path = 'C:\\Users\\Simon\\Desktop'
os.chdir(path)
img = imread("UZH_9_low_inter_area.png")
img_old = imread("UZH_9_mask_low.png")
count = img == 128
sum(sum(count))



# check for wrong pixels
f = os.listdir()
files = f[900:1400]
for file in files:
    img = imread(file)
    count = img == 1 # or img == 126
    s = sum(sum(count))
    if sum(s) > 0:
        print(s)
        print(file)

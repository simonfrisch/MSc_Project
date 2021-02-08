import os
import cv2
import imageio

path = "D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\DD_augmented_cropped\\water_unsharped"

def img_resize(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        image = cv2.imread(file)
        resized_image = image[0:473, 0:473] # crop image
        #resized_image = cv2.resize(image, (640, 480)) # resize image
        name = file
        cv2.imwrite(name, resized_image)

water = [0, 0, 0]
nonwater = [255, 255, 255]
water_ratio = []
im_width = 160
im_height = 120

def check_water_ratio(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        water_px = 0
        nonwater_px = 0
        binarymask = imageio.imread(file)
        for p in range(im_height):
            for q in range(im_width):
                if((binarymask[p, q] == water).all()):
                    water_px += 1
                elif((binarymask[p, q] == nonwater).all()):
                    nonwater_px += 1
        ratio = water_px/(water_px+nonwater_px)
        water_ratio.append(ratio)
        print(file)
    with open("ratio.txt", "w") as output:
        output.write(str(water_ratio))

img_resize(path)

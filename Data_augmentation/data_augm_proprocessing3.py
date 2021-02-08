import os
import cv2
#os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\tugraz_augmented\\water_gaussian_filter")
nr = os.listdir()

path = "D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\DD_augmented\\water_unsharped"

def copy_image(path):
    os.chdir(path)
    files = os.listdir()
    j = 0
    for i in range(len(files)):
        if nr[j] == files[i]:
            img = cv2.imread(files[i])
            img_name = "_" + files[i]
            cv2.imwrite(img_name, img)
            if j <= len(nr) - 1:
                j += 1

def copy_image_simple(path):
    os.chdir(path)
    files = os.listdir()
    for file in files:
        image = cv2.imread(file)
        splt = file.split("_")
        splt2 = splt[0] + "_" + splt[1] + "_augmented_" + splt[3] # + "_" + splt[4]
        name = splt2
        cv2.imwrite(name, image)

copy_image_simple(path)

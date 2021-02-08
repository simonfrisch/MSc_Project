import os
import cv2

# original images
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\semantic_drone_dataset_renamed\\Downsampled_images\\images_downsampled")
files = os.listdir()

for file in files:
    image = cv2.imread(file)
    name = 'tugraz_' + file[8:11] + "_original" + ".jpg"
    cv2.imwrite(name, image)

# maks (labels)
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\semantic_drone_dataset_renamed\\Downsampled_images\\label_images_downsampled")
files = os.listdir()

for file in files:
    image = cv2.imread(file)
    name = 'tugraz_' + file[18:21] + "_mask" + ".png"
    cv2.imwrite(name, image)

# binary masks (black-white)
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\semantic_drone_dataset_renamed\\Downsampled_images\\masks")
files = os.listdir()

for file in files:
    image = cv2.imread(file)
    name = 'tugraz_' + file[18:21] + "_binmask" + ".png"
    cv2.imwrite(name, image)

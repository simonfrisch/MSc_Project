import os
import cv2
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\Drone_Deploy_renamed")

lines = []
with open('ratio.txt') as infile:
    file_lines = infile.read().splitlines()

for line in file_lines:
    lines.append([float(v) for v in line[1:-1].split(',')])

w = lines[0]

img_nr = [i > 0.0001 for i in w]
nr = []
for i in range(len(w)):
    if img_nr[i] == True:
        nr.append(i)

path = "D:\\Documents\\UZH\\FS20\\MSc Project\\Daten\\Drone_Deploy_renamed\\images_downsampled"

def copy_image(path):
    os.chdir(path)
    files = os.listdir()
    j = 0
    for i in range(len(w)):
        if nr[j] == i:
            img = cv2.imread(files[i])
            img_name = "_" + files[i]
            cv2.imwrite(img_name, img)
            if j <= len(nr) - 2:
                j += 1

copy_image(path)

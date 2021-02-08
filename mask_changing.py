import cv2
import numpy as np
import os
colordict = {"building": [0, 0, 255], "vehicle": [0, 127, 255], "road": [80, 80, 80], "gravel": [255, 127, 255], "water": [255, 0, 0], \
                          "person": [0, 255, 255], "tree": [0, 127, 0], "pavedarea": [220, 220, 220], "rooftop": [127, 0, 127], "grass": [0, 255, 0], "other": [0, 0, 0]}
pathdataset = "C:/Users/Simon/Downloads/"
newpath = "C:/Users/Simon/Downloads/masks_corr/"
maskname = "UZH_347_mask.png"
mask = cv2.imread(pathdataset + maskname)
newmask = mask.copy()
newmask[np.where((newmask == colordict["tree"]).all(axis=2))] = colordict["grass"]
partly = True
im_height = 3000
im_width = 4000
if partly:
    vstart = int(0 * im_height)
    vend = int(1 * im_height)
    hstart = int(0.24 * im_width)
    hend = int(0.6 * im_width)
    mask[vstart:vend,hstart:hend] = newmask[vstart:vend,hstart:hend]
    print("writing image", newpath + maskname)
    cv2.imwrite(newpath + maskname, mask)
else:
    print("writing image", newpath + maskname)
    cv2.imwrite(newpath + maskname, newmask)


import sys
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages') # to add path where packages are installed
sys.path.insert(0,'/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/bin') # to add path where packages are installed
import cv2 as cv
from os import walk
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import tensorflow as tf
from keras.optimizers import Adam
from PIL import Image
import segmentation_models as sm

themodel = "model_19_masks10"

number_of_classes = 10

pathtestset = "/Users/alexschindler/Documents/_Studium/2020_FS/Masterprojektarbeit/Dataset_05_09/Test/"

pathmodel = "/Users/alexschindler/Documents/_Studium/2020_FS/Masterprojektarbeit/rpg_master_project/models/trained_models/"

pathpredictions = pathtestset + themodel + "_Predicted/"

#make folder to output predicted images
if os.path.exists(pathpredictions):
    shutil.rmtree(pathpredictions)
os.makedirs(pathpredictions)

# Set some parameters
im_width = 640
im_height = 480

# Get test images (and masks - though the masks do not get used here)
def get_data(path):
    print("number 1")
    os.chdir(path)

    pathoriginal = path + "Original/"
    pathbinarymasks = path + "Binary_masks/"
    for (dirpath, dirnames, filenames) in walk(pathoriginal):
        x_l = []
        mask_l = []
        pictureprefixes = []
        picturenumbers = []
        for filename in filenames:
            if (filename[-3:] != "png"):  # do not consider non-png or jpg-files
                if (filename[-3:] != "jpg"):
                    continue

            print('loaded image: ' + filename)
            img = cv.imread(pathoriginal + filename)

            x_l.append(img.squeeze() / 255)
            # Load masks
            nameparts = filename.split('_')
            image_prefix = nameparts[0]
            image_id = int(nameparts[1])
            pictureprefixes.append(image_prefix)
            picturenumbers.append(int(image_id))
            mask_name = image_prefix + '_' + str(image_id) + '_binarymask.png'
            print('loaded mask: ' + mask_name)
            mask = cv.imread(pathbinarymasks + mask_name, cv.IMREAD_GRAYSCALE)
            mask_l.append(mask / 255)

        X = np.zeros((len(x_l), im_height, im_width, 3), dtype=np.float32)
        y = np.zeros((len(mask_l), im_height, im_width, 1), dtype=np.float32)
        print("y shape", y.shape)
        X = np.array(x_l)
        y = np.array(mask_l)
        y = y[..., np.newaxis] # this adds a dimension at the end
        print("y shape", y.shape)
        return X, y, picturenumbers, pictureprefixes

#load the pretrained unet
model = sm.Unet()
model = sm.Unet('resnet34', classes=number_of_classes, activation='softmax', encoder_weights='imagenet')

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("about to load", pathmodel + themodel + ".h5")
model.load_weights(pathmodel + themodel + ".h5")

# get the testdata
X_test, Y_test, picnumbers, picprefixes = get_data(pathtestset)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

#get predictions for the images
predictions = model.predict(X_test);

def create_mask(pred_mask): # gets the index of the entry with the maximum prediction value
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

#colors for classes: other, water, road, building, vehicle, gravel, person, tree, rooftop, grass, pavedarea
colormap = np.array([[0, 0, 0], [255, 0, 255],  [0, 0, 127], [255, 0, 0], [0, 0, 255], [127, 0, 0], \
              [255, 0, 127], [0, 255, 0], [127, 0, 127], [255, 255, 0]])
               # "pavedarea": [0, 0, 127],

for x in range(len(predictions)): # go through predictions, turn them into images, save them
    pred_mask = create_mask(predictions[x]) # get the index of the maximum entry

    #use colormap to get the right pixel color
    pred_image = colormap[pred_mask[:,:, -1]]
    pred_image_reshaped = np.array(pred_image).astype(np.uint8)

    #turn array into image
    im = Image.fromarray(pred_image_reshaped)

    #get new filename
    filename = pathpredictions + picprefixes[x] + '_' + str(picnumbers[x]) + '_' + themodel + 'prediction.png'
    print("saving image", filename)
    im.save(filename)
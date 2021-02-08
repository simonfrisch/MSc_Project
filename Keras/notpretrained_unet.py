import sys
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages') # to add path where packages are installed
sys.path.insert(0,'/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/bin') # to add path where packages are installed

import os
import random
import math
import numpy as np

from itertools import chain
import tensorflow as tf
import io
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras import backend as K


# Notes:
# Using pretrained Unet from segmentation_models
# Lossfunction: Adjusted Categorical Crossentropy to ignore unlabeled pixels
# Activation: Softmax

# Set some parameters
modelname = 'model_23_low_augmented_notpretrained_unet.h5'

if("low" in modelname):
    resolution = "low"
elif("high" in modelname):
    resolution = "high"
if(resolution == "high"):
    im_width = 640
    im_height = 480
elif(resolution == "low"):
    im_width = 480
    im_height = 320
path_train = "/Users/alexschindler/Documents/_Studium/2020_FS/Masterprojektarbeit/Dataset_06_02/Train/"
data = 'Batch1_Original_high'
datasets = ['UZH']
testsetnumbers = [150, 450, 41, 36, 117, 217, 198, 321, 25, 4, 370, 336, 363, 220, 52, 285, 199, 264, 245, 192, 302, 22, 185, 17, 175, 377, 252, 298, 115, 169, 78, 1, 207, 80, 295, 428, 342, 85, 372, 326, 392, 402, 433, 397, 382, 460, 89, 98, 373, 288]
percentage_of_val_set = 0.15

colordict = {"building": [0, 0, 255], "vehicle": [0, 127, 255], "road": [80, 80, 80], "gravel": [255, 127, 255],
             "water": [255, 0, 0], \
             "person": [0, 255, 255], "tree": [0, 127, 0], "pavedarea": [220, 220, 220], "rooftop": [127, 0, 127],
             "grass": [0, 255, 0], \
             "other": [0, 0, 0]}

#put classes that should go together in a list
output = [["water"],
          ["building", "vehicle", "road", "gravel", "person", "tree", "pavedarea", "rooftop", "grass", "other"]]

outputdict = {}
for k in range(len(output)):
    for c in output[k]:
        outputdict[c] = k

print(outputdict)

number_of_classes = len(output)



def get_image_names(path, data):
    image_names = []
    ids = next(os.walk(path + data))[2]
    for id in ids:
      split = id.split('_')
      dataset_id = split[0]
      number = split[1]
      if dataset_id in datasets and number not in testsetnumbers and int(number) < 1000:
        image_names.append(id)
    return image_names

image_names = get_image_names(path_train, data)
#print(image_names)

random.shuffle(image_names)

print(image_names[0:10])

train_image_names = image_names[0:math.floor(len(image_names)*(1-percentage_of_val_set))]
validation_image_names = image_names[math.floor(len(image_names)*(1-percentage_of_val_set)):len(image_names)]




# get the effective images and the masks
def get_data(filenames, path):
    x_l = []
    mask_l = []
    X = np.zeros((len(filenames), im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(filenames), im_height, im_width, number_of_classes), dtype=np.float32)
    # Load images and masks
    for n, filename in enumerate(filenames):
      #print("filename", filename)
      # load images
      img = load_img(path + data + '/' + filename, grayscale=False)
      x_img = img_to_array(img)
      X[n, ...] = x_img.squeeze() / 255

      split = filename.split('_')
      if '_rotate' in filename:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '_rotate_categorical10.png'
      elif '_cropped' in filename:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '_cropped_categorical10.png'
      elif '_dropout' in filename:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '_dropout_categorical10.png'
      elif '_flipped' in filename:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '_flipped_categorical10.png'
      elif '_remove_row' in filename:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '_remove_row_categorical10.png'
      else:
        mask_name = split[0] + '_' + split[1] + '_mask_' + resolution + '.png'
      mask = img_to_array(load_img(path + 'Batch1_Masks_' + resolution + '/' + mask_name))

      # transform masks (masks need to be in format w x h x c, where c is the index of a category e.q. 0,1,2 etc.
      newmask = np.zeros((im_height, im_width, 1), dtype="uint8")
      for color in colordict:
          newmask[np.where((mask == colordict[color]).all(axis=2))] = outputdict[color]

      # create hot one vector matrices from the category indices
      y[n] = to_categorical(newmask, num_classes=number_of_classes, dtype='float32')
    return X, y

imagearray, maskarray = get_data(image_names, path_train)
print("image", imagearray[0])
print("masks", maskarray[0])



def imageLoader(files, batch_size):
    L = len(files)
    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = get_data(files[batch_start:limit], path_train)
            # print("X", X[0][0][0], "Y", Y[0][0][0])

            batch_start += batch_size
            batch_end += batch_size

            yield (X,Y)


def get_unet(input_img):
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (input_img)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(number_of_classes, (1, 1), activation='softmax') (c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((im_height, im_width, 3), name='img')

model = get_unet(input_img)

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.load_weights(path_train + 'model_23_augmented_low_notpretrained_unet.h5')


class EpochEndCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    random.shuffle(train_image_names)
    random.shuffle(validation_image_names)

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(path_train + modelname, verbose=1, save_best_only=True, save_weights_only=True),
    EpochEndCallback()
]
batch_size = 32

results = model.fit_generator(imageLoader(train_image_names,batch_size),
                              steps_per_epoch=len(train_image_names)//batch_size,
                              validation_data = imageLoader(validation_image_names, batch_size),
                              validation_steps = len(validation_image_names)//batch_size,
                              epochs=100,
                              callbacks=callbacks,
                              shuffle=True)
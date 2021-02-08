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

import segmentation_models as sm

# Notes:
# Using pretrained Unet from segmentation_models
# Lossfunction: Adjusted Categorical Crossentropy to ignore unlabeled pixels
# Activation: Softmax

# Set some parameters
im_width = 640
im_height = 480
path_train = '/data/scratch/rstaubli/Dataset_05_15/Train/'
data = 'Augmented'
datasets = ['tugraz','Zurich','Aargau','DD']
percentage_of_val_set = 0.15
number_of_classes = 3

def get_image_names(path, data):
    image_names = []
    ids = next(os.walk(path + data))[2]
    for id in ids:
      dataset_id = id.split('_')[0]
      if dataset_id in datasets:
        image_names.append(id)
    return image_names

image_names = get_image_names(path_train, data)

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
        mask_name = split[0] + '_' + split[1] + '_mask2categorical_rotate.png'
      else:
      	mask_name = split[0] + '_' + split[1] + '_mask2categorical.png'
      # load masks (masks need to be in format w x h x c, where c is the index of a category e.q. 0,1,2 etc.
      mask = img_to_array(load_img(path + 'Masks2categorical/' + mask_name, grayscale=True))
      # create hot one vector matrices from the category indices
      y[n] = tf.keras.utils.to_categorical(mask, num_classes=number_of_classes, dtype='float32')
    return X, y

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

model = sm.Unet()
model = sm.Unet('resnet34', classes=number_of_classes, activation='softmax', encoder_weights='imagenet')

# Ignore the last category in the loss by setting the value to 0 for lables that are not labeled
def masked_loss_function(y_true, y_pred):
  mask_value=tf.constant([1.0,1.0,0.0], dtype=tf.float32)
  y_true_masked = mask_value * y_true
  y_pred_masked = mask_value * y_pred
  return K.categorical_crossentropy(y_true_masked, y_pred_masked)

model.compile(optimizer=Adam(), loss=masked_loss_function, metrics=['accuracy'])
model.summary()

class EpochEndCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    random.shuffle(train_image_names)
    random.shuffle(validation_image_names)

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(path_train + 'model_20_masks2.h5', verbose=1, save_best_only=True, save_weights_only=True),
    EpochEndCallback()
]
batch_size = 16

results = model.fit_generator(imageLoader(train_image_names,batch_size),
                              steps_per_epoch=len(train_image_names)//batch_size,
                              validation_data = imageLoader(validation_image_names, batch_size),
                              validation_steps = len(validation_image_names)//batch_size,
                              epochs=100,
                              callbacks=callbacks,
                              shuffle=True)

# for dataset Drone Deploy
import sys
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages') # to add path where packages are installed
sys.path.insert(0,'/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/bin') # to add path where packages are installed
import cv2 as cv

from os import walk
import os
import shutil
import numpy as np
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use("ggplot")

from skimage.io import imread, imshow, concatenate_images, imsave

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from PIL import Image

themodel = "model_74_high_notpretrained_unet"

number_of_classes = 6

minpictureid = 1 # to not run the whole data set
maxpictureid = 1500

#for k in range(20):

predict = True
IoU = True
cutuppredictstitchtogether = False
video = False
videonr = "UZH"

if (number_of_classes == 2):
    colormap = np.array([[0, 0, 255], [255, 255, 255]])  # water: blue, other: white
elif (number_of_classes == 3):
    colormap = np.array([[0, 0, 255], [80, 80, 80], [255, 255, 255]])  # water: black; road: gray; other: white
elif (number_of_classes == 4):
    colormap = np.array([[0, 0, 255], [80, 80, 80], [0, 255, 0], [255, 255, 255]])  # water: blue; road: gray; grass: light green;  grass: light green; other: white
    #colormap = np.array([[0, 127, 0], [80, 80, 80], [0, 255, 0], [255, 255, 255]])  # tree: dark green; road: gray; grass: light green; other: white
elif (number_of_classes == 5):
    colormap = np.array([[0, 0, 255], [80, 80, 80], [0, 255, 0], [0, 127, 0], [255, 255, 255]])  # water: blue, grass: light green; road: gray; grass: light green; tree: dark green; other: white
elif (number_of_classes == 6):
    colormap = np.array([[0, 0, 255], [80, 80, 80], [0, 255, 0], [0, 127, 0], [127, 0, 127], [255, 255, 255]])  # water: blue, grass: light green; road: gray; grass: light green; tree: dark green; rooftop: purple; other: white

elif (number_of_classes == 10):
    colormap = np.array([[0, 0, 0], [255, 0, 255], [0, 0, 127], [255, 0, 0], [0, 0, 255], [127, 0, 0], \
                         [255, 0, 127], [0, 255, 0], [127, 0, 127], [255, 255, 0]]) # "pavedarea": [0, 0, 127],
elif (number_of_classes == 11):

    #  {"building": [0, 0, 255], "vehicle": [0, 127, 255], "road": [80, 80, 80], "gravel": [255, 127, 255],
    #   "water": [255, 0, 0], \
    #   "person": [0, 255, 255], "tree": [0, 127, 0], "pavedarea": [220, 220, 220],
    #   "rooftop": [127, 0, 127], "grass": [0, 255, 0], \
    #   "other": [0, 0, 0]}
    #  {"other": [0, 0, 0], "water": [1, 1, 1], "road": [2, 2, 2], "building": [3, 3, 3],
    #   "vehicle": [4, 4, 4], \
    #  "gravel": [5, 5, 5], "person": [6, 6, 6], "tree": [7, 7, 7], \
    #  "rooftop": [8, 8, 8], "grass": [9, 9, 9], "pavedarea": [10, 10, 10]}

    # colormap = np.array([[0, 0, 0], [255, 0, 0], [80, 80, 80], [0, 0, 255], [0, 127, 255], [255, 127, 255], \
    #                     [0, 255, 255], [0, 127, 0], [127, 0, 127], [0, 255, 0], [220, 220, 220]])
    colormap = np.array([[0, 0, 0], [0, 0, 255], [80, 80, 80], [255, 0, 0], [255, 127, 0], [255, 127, 255], \
                         [255, 255, 0], [0, 127, 0], [127, 0, 127], [0, 255, 0], [220, 220, 220]])


testsetnumberstugraz = [15, 31, 40, 42, 47, 53, 55, 60, 79, 81, 83, 112, 122, 126, 136, 145, 146, 154, 158, 163, 176, 185,
                        190, 192, 202, 213, 214, 215, 250, 255, 275, 295, 322, 329,
                        351, 385, 390, 410, 413, 427, 461, 467, 479, 500, 507, 530, 531, 554, 564, 585]

testsetnumbersUZH = [1, 4, 8, 21, 30, 51, 62, 76, 86, 96, 101, 112, 114, 118, 133, 135, 162, 167, 187, 192, 194,
                     216, 218, 222, 225, 226, 238, 252, 256, 259, 270, 275, 283, 285, 289, 295, 303, 304,
                     308, 309, 317, 320, 324, 332, 337, 350, 355, 376, 392, 398, 405, 413, 429, 437, 439, 442,
                     443, 445, 456, 460, 474, 478, 486, 497, 502, 508, 526, 539, 541, 546, 553, 576, 581, 582, 588,
                     592, 597, 605, 629, 630, 651, 656, 658, 659, 660, 667, 670, 671, 682, 690, 702, 706, 718,
                     720, 721, 731, 737, 749, 764, 767, 781, 795, 803, 806, 824, 829, 836, 879, 890, 891, 894,
                     898, 922, 929, 930, 931, 940, 944, 954, 959, 965, 974, 986, 989, 993]


pathdataset = "C:/Users/Simon/Downloads/Testing/"

pathmodel = "C:/Users/Simon/Downloads/Testing/"


if cutuppredictstitchtogether:
    pathpredictions = pathdataset + "Test/" + themodel + "_Predictions_cutup/"
elif video:
    pathpredictions = pathdataset + "Videos/" + themodel + "_" + str(videonr) + "_Predictions/"
else:
    pathpredictions = pathdataset + "Test/" + themodel + "_Predictions/"

if predict:
#    while(1):
    if not os.path.exists(pathpredictions):
#            pathpredictions = pathpredictions[:-1] + "_new/"
#        else:
#            break
#
        os.makedirs(pathpredictions)

if("high" in themodel):
    resolution = "high"
elif("low" in themodel):
    resolution = "low"
elif("huge" in themodel):
    resolution = "huge"

if(resolution == "high"):
    im_width = 640
    im_height = 480
elif(resolution == "low"):
    im_width = 480
    im_height = 320
elif(resolution == "huge"):
    im_width = 2560
    im_height = 1920

original_width = 4000
original_height = 3000

if cutuppredictstitchtogether:
    def makecutupfilenames(path, testsetname, testset, defaultnrcuts):
        filenames = []
        for k in range(1250):  # maximal image ID
            originalfilename = path + "Original_originalsize/" + testsetname + "_" + str(k) + "_original.jpg"
            print(originalfilename)
            if not os.path.isfile(originalfilename):
                print("image with number", k, "does not exist")
                continue

            if (k not in testset):
                continue

            if(minpictureid < k or k > maxpictureid):
                continue


            if defaultnrcuts > 0:
                nrcuts = defaultnrcuts
            # randomize the number of images the original image is cut up in
            else:
                nrcuts = randint(minimages, maximages)
            for cutnr in range(nrcuts ** 2):
                cutnr += 1
                filename = testsetname + "_" + str(k) + "_" + str(nrcuts) + "_" + str(cutnr)
                if os.path.exists(path + "Original_cut/" + filename + "_original_cut.png"):
                    print("file", filename, "already exists")
                    continue
                filenames.append(filename)
        print("number of pictures:", len(filenames))
        return filenames

    def makecutupimages(path, filenames):
        for filename in filenames:
            filenameparts = filename.split("_")
            origin = filenameparts[0]
            image_id = filenameparts[1]
            nrcuts = int(filenameparts[2])  # number of images the original gets cut into (in each direction)
            cutnr = int(filenameparts[3])  # which of the cut up images is it?
            print(path + "Original_originalsize/" + origin + "_" + image_id + "_original.jpg")
            originalimage = cv.imread(path + "Original_originalsize/" + origin + "_" + image_id + "_original.jpg")
            #originalmask = cv.imread(path + "Masks_originalsize/" + origin + "_" + image_id + "_mask.png")

            tempheight = original_height // nrcuts  # temporary height and width (before downsampling)
            tempwidth = original_width // nrcuts
            x = (cutnr - 1) // nrcuts
            y = (cutnr - 1) % nrcuts
            print(cutnr, x, y)

            tempimage = originalimage[x * tempheight: (x + 1) * tempheight, y * tempwidth: (y + 1) * tempwidth]
            #tempmask = originalmask[x * tempheight: (x + 1) * tempheight, y * tempwidth: (y + 1) * tempwidth]

            cutupimage = cv.resize(tempimage, (im_width, im_height), interpolation=cv.INTER_NEAREST)
            #cutupmask = cv.resize(tempmask, (im_width, im_height), interpolation=cv.INTER_NEAREST)

            newfilenameimage = path + "Original_cut/" + origin + "_" + str(image_id) + "_" + str(nrcuts) + "_" + str(cutnr) + "_original_cut.png"
            #newfilenamemask = path + "Masks_cut/" + origin + "_" + str(image_id) + "_" + str(nrcuts) + "_" + str(
            #    cutnr) + "_mask_cut.png"

            cv.imwrite(newfilenameimage, cutupimage)
            #cv.imwrite(newfilenamemask, cutupmask)

    #create the filenames
    cutupimagesfilenames = makecutupfilenames(pathdataset, "UZH", testsetnumbersUZH, 6) + makecutupfilenames(pathdataset, "tugraz", testsetnumberstugraz, 6)

    #make the images
    makecutupimages(pathdataset, cutupimagesfilenames)


if predict:
    def get_data(path, testsetnumbersUZH, testsetnumberstugraz):
        # Get test images
        print("Loading images ...")
        #os.chdir(path)


        if(resolution == "high"):
            if cutuppredictstitchtogether:
                pathoriginal = path + "Original_cut/"
            elif video:
                pathoriginal = path + "Videos/" + str(videonr) + "_original_high/"
            else:
                pathoriginal = path + "Original_high/"
        elif(resolution == "low"):
            pathoriginal = path + "Original_low/"
        elif (resolution == "huge"):
            pathoriginal = path + "Original_originalsize/"

        print("Test images in path", pathoriginal)

        for (dirpath, dirnames, filenames) in walk(pathoriginal):
            print("Creating image array")
            x_l = []
            pictureprefixes = []
            picturenumbers = []
            for filename in filenames:
                print("working on file", filename)
                if (filename[-3:] != "png"):  # do not consider non-png or jpg-files
                    if (filename[-3:] != "jpg"):
                        continue

                # Load masks
                nameparts = filename.split('_')
                print(filename)
                image_prefix = nameparts[0]
                image_id = int(nameparts[1])
                if cutuppredictstitchtogether:
                    image_id2 = int(nameparts[2])
                    image_id3 = int(nameparts[3])
                    fullid = str(image_id) + "_" + str(image_id2) + "_" + str(image_id3)

                if(image_id < minpictureid or image_id > maxpictureid):
                    continue
                if not video:
                    if(image_prefix == "UZH" and int(image_id) not in testsetnumbersUZH):
                        continue
                    if (image_prefix == "tugraz" and image_id not in testsetnumberstugraz):
                        continue

                print('Loading image: ' + filename)
                img = cv.imread(pathoriginal + filename)
                if(resolution == "huge"):
                    img = cv.resize(img, (im_width, im_height), interpolation=cv.INTER_NEAREST)

                if cutuppredictstitchtogether:
                    picturenumbers.append(fullid)
                else:
                    picturenumbers.append(int(image_id))
                pictureprefixes.append(image_prefix)

                x_l.append(img.squeeze() / 255)

            X = np.zeros((len(x_l), im_height, im_width, 3), dtype=np.float32)
            X = np.array(x_l)
            return X, picturenumbers, pictureprefixes

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


    print("Loading UNET")
    input_img = Input((im_height, im_width, 3), name='img')
    model2 = get_unet(input_img)

    model2.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    model2.summary()

    print("Loading model", pathmodel + themodel + ".h5")
    model2.load_weights(pathmodel + themodel + ".h5")


    print("Loading images ...")
    X_test, picnumbers, picprefixes = get_data(pathdataset, testsetnumbersUZH, testsetnumberstugraz)

    print("X_test shape:", X_test.shape)

    predictions = model2.predict(X_test);


    def create_mask(pred_mask):  # gets the index of the entry with the maximum prediction value
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask


    # go through predictions, turn them into images, save them
    for x in range(len(predictions)):
    #for x in range(0):
        if (x < 3):
            print(predictions[x])
        pred_mask = create_mask(predictions[x])  # get the index of the maximum entry
        if (x < 3):
            print(pred_mask)
        print(pred_mask.shape)
        pred_image = colormap[pred_mask[:, :, -1]]

        pred_image_reshaped = np.array(pred_image).astype(np.uint8)

        im = Image.fromarray(pred_image_reshaped)
        filename = pathpredictions + picprefixes[x] + '_' + str(picnumbers[x]) + '_' + themodel + '_prediction.png'
        im.save(filename)


if cutuppredictstitchtogether:

    if not os.path.exists(pathpredictions[:-1] + "_stitched"):
        os.makedirs(pathpredictions[:-1] + "_stitched")
    def stitchtogether(path, model, datasetname, filenames):
        nrcuts = 6
        for nr in filenames:
            if(int(nr) > maxpictureid or int(nr) < minpictureid):
                continue
            print(nr)
            largeimage = np.zeros((im_height * nrcuts, im_width * nrcuts, 3), dtype="uint8")
            # largemask = np.zeros((im_height*nrcuts, im_width*nrcuts,3), dtype="uint8")
            # tempheight = original_height // nrcuts  # temporary height and width (before downsampling)
            # tempwidth = original_width // nrcuts
            for cutnr in range(36):
                cutnr += 1

                filename = pathpredictions + datasetname + "_" + str(nr) + "_" + str(nrcuts) + "_" + str(cutnr) + "_" + model + "_prediction.png"
                # filename = path + "Test/Original_cut" + "/UZH_" + str(nr) + "_" + str(nrcuts) + "_" + str(cutnr) + "_original_cut.png"
                print(filename)
                image = cv.imread(filename)
                x = (cutnr - 1) // nrcuts
                y = (cutnr - 1) % nrcuts
                print(cutnr, x, y)

                largeimage[x * im_height: (x + 1) * im_height, y * im_width: (y + 1) * im_width] = image
                # largemask[x * tempheight: (x + 1) * tempheight, y * tempwidth: (y + 1) * tempwidth] = mask
            stitchedtogetherdownscaled = cv.resize(largeimage, (im_width, im_height), interpolation=cv.INTER_NEAREST)
            newfilename = pathpredictions[:-1] + "_stitched/" + datasetname + "_" + str(nr) + "_" + model + "_prediction.png"
            print("new filename", newfilename)
            cv.imwrite(newfilename, stitchedtogetherdownscaled)


    stitchtogether(pathdataset, themodel, "UZH", testsetnumbersUZH)
    stitchtogether(pathdataset, themodel, "tugraz", testsetnumberstugraz)

#minpictureid+=50
#maxpictureid+=50

def performanceIoU():


    cols = ['Picture name']
    for cat in categories:
        cols = cols + [cat + '_IoU'] + [cat + '_groundtruth_pixels'] + [cat + '_prediction_pixels'] + [cat + '_weighted_IoU_groundtruth'] + [cat + '_weighted_IoU_prediction']
    print("columns", len(cols))
    performanceall = pd.DataFrame(columns=cols)
    performancetugraz = pd.DataFrame(columns=cols)
    performanceUZH = pd.DataFrame(columns=cols)
    print(performanceall)

    #os.chdir(pathtestset)
    if(cutuppredictstitchtogether):
        files = os.listdir(pathpredictions[:-1] + "_stitched/")
    else:
        files = os.listdir(pathpredictions)
    print(files)
    k = 0
    i = 0
    j = 0
    for filename in files:
        #print(k)
        if (filename[-3:] != "png"):  # do not consider non-png-files
            if (filename[-3:] != "jpg"):  # do not consider non-jpg-files
                if (filename[-3:] != "JPG"):
                    if (filename[-3:] != "PNG"):
                        continue
        print(filename)
        nameparts = filename.split('_')

        origin = nameparts[0]
        pictureid = int(nameparts[1])

        #if pictureid > 20:
        #    continue

        if "high" in filename:
            maskfilename = "Masks_high/" + origin + "_" + str(pictureid)+ "_mask_high.png"
        elif "low" in filename:
            maskfilename = "Masks_low/" + origin + "_" + str(pictureid) + "_mask_low.png"
        elif "huge" in filename:
            maskfilename = "Masks_originalsize/" + origin + "_" + str(pictureid) + "_mask.png"
        if cutuppredictstitchtogether:
            print("Prediction", pathpredictions[:-1] + "_stitched/" + filename)
            predictedimage = cv.imread(pathpredictions[:-1] + "_stitched/" + filename)
        else:
            print("Prediction", pathpredictions + filename)
            predictedimage = cv.imread(pathpredictions + filename)
        print("mask filename", pathdataset + maskfilename)
        mask = cv.imread(pathdataset + maskfilename)
        if (resolution == "huge"):
            mask = cv.resize(mask, (im_width, im_height), interpolation=cv.INTER_NEAREST)
        newmask = np.zeros((im_height, im_width, 3), dtype="uint8")


        iou = [origin + "_" + str(pictureid)]
        for category in categories:
            for cat in output[category]:
                newmask[np.where((mask == colordict_original[cat]).all(axis=2))] = outputdict[cat]
        for category in categories:
            groundtruthpixels = np.count_nonzero((newmask == colordict_model[category]).all(axis=2))
            predictionpixels = np.count_nonzero((predictedimage == colordict_model[category]).all(axis=2))
            intersection = np.count_nonzero(np.logical_and((predictedimage == colordict_model[category]).all(axis=2), (newmask == colordict_model[category]).all(axis=2)))
            union = np.count_nonzero(np.logical_or((predictedimage == colordict_model[category]).all(axis=2), (newmask == colordict_model[category]).all(axis=2)))

            if(union > 0):
                iou.append(intersection /union)
                iou.append(groundtruthpixels)
                iou.append(predictionpixels)
                iou.append(groundtruthpixels * intersection / union)
                iou.append(predictionpixels * intersection / union)
            else:
                iou.append(np.NaN)
                iou.append(groundtruthpixels)
                iou.append(predictionpixels)
                iou.append(0)
                iou.append(0)

        print(iou)

        if(origin == "tugraz"):
            performancetugraz.loc[i] = iou
            i+=1
        elif(origin == "UZH"):
            performanceUZH.loc[j] = iou
            j+=1
        performanceall.loc[k] = iou
        k += 1
    mean = []
    for c in cols:
        if(c == 'Picture name'):
            mean.append('mean')
            continue
        mean.append(performanceUZH[c].mean())#(numeric_only=True))
    print(mean)
    performanceUZH.loc['mean'] = mean  # ['Picture name'] = "mean" #= performanceUZH.mean(numeric_only=True)


    sum = []
    for c in cols:
        if(c == 'Picture name'):
            sum.append('sum')
            continue
        sum.append(performanceUZH[c].sum() - performanceUZH.loc['mean'][c])

    print(sum)
    performanceUZH.loc['sum'] = sum  # ['Picture name'] = "mean" #= performanceUZH.mean(numeric_only=True)


    weighted = []
    for c in cols:
        if(c == 'Picture name'):
            weighted.append('weighted')
            continue
        colparts = c.split("_")

        if(colparts[1] == "weighted"):
            if(performanceUZH.loc['sum'][colparts[0] + "_" + colparts[3] + "_" + "pixels"] == 0):
                weighted.append(np.NaN)
                continue
            weighted.append(performanceUZH.loc['sum'][c] / performanceUZH.loc['sum'][colparts[0] + "_" + colparts[3] + "_" + "pixels"])
        else:
            weighted.append(np.NaN)
    print(weighted)
    performanceUZH.loc['weighted'] = weighted

    #performanceUZH.loc['sum'] = performanceUZH.sum(numeric_only=True)
    performancetugraz.loc['mean'] = performancetugraz.mean(numeric_only=True)
    performanceall.loc['mean'] = performanceall.mean(numeric_only=True)
    filenameperformanceall =  pathdataset + "Test/" + themodel + "_iou_all.xlsx"
    if cutuppredictstitchtogether:
        filenameperformanceUZH = pathdataset + "Test/" + themodel + "_stitched_iou_UZH.xlsx"
    else:
        filenameperformanceUZH = pathdataset + "Test/" + themodel + "_iou_UZH.xlsx"
    filenameperformancetugraz = pathdataset + "Test/" + themodel + "_iou_tugraz.xlsx"
    performanceall.to_excel(filenameperformanceall, index=False)
    performanceUZH.to_excel(filenameperformanceUZH, index=False)
    performancetugraz.to_excel(filenameperformancetugraz, index=False)

#put classes that should go together in a list
if number_of_classes == 2:
    output = {"water" : ["water"],
          "other" : ["building", "vehicle", "road", "gravel", "person", "tree", "pavedarea", "rooftop", "grass", "other"]}
    colordict_model = {"water": [255, 0, 0], "other": [255, 255, 255]}

elif number_of_classes == 3:
    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea", "gravel"],
          "other" : ["building", "vehicle", "person", "tree", "rooftop", "grass", "other"]}
    colordict_model = {"water" : [255, 0, 0], "roadpavedarea": [80, 80, 80], "other": [255, 255, 255]}
	
#elif number_of_classes == 3:
#    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea"],
#          "other" : ["building", "vehicle", "gravel", "person", "tree", "rooftop", "grass", "other"]}
#    colordict_model = {"water" : [255, 0, 0], "roadpavedarea": [80, 80, 80], "other": [255, 255, 255]}

elif number_of_classes == 4:
    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea", "gravel"], "grass" : ["grass"],
          "other" : ["building", "vehicle", "person", "tree", "rooftop", "other"]}
    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "other": [255, 255, 255]}

#elif number_of_classes == 4:
#    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea"], "grass" : ["grass"],
#          "other" : ["building", "vehicle", "gravel", "person", "tree", "rooftop", "other"]}
#    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "other": [255, 255, 255]}

elif number_of_classes == 5:
    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea", "gravel"], "grass" : ["grass"], "tree" : ["tree"],
          "other" : ["building", "vehicle", "person", "rooftop", "other"]}
    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "tree" : [0, 127, 0], "other": [255, 255, 255]}

#elif number_of_classes == 5:
#    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea"], "grass" : ["grass"], "tree" : ["tree"],
#          "other" : ["building", "vehicle", "gravel", "person", "rooftop", "other"]}
#    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "tree" : [0, 127, 0], "other": [255, 255, 255]}

elif number_of_classes == 6:
    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea", "gravel"], "grass" : ["grass"], "tree" : ["tree"], "rooftop" : ["rooftop"],
          "other" : ["building", "vehicle", "person", "other"]}
    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "tree" : [0, 127, 0], "rooftop" : [127, 0, 127], "other": [255, 255, 255]}

#elif number_of_classes == 6:
#    output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea"], "grass" : ["grass"], "tree" : ["tree"], "rooftop" : ["rooftop"],
#          "other" : ["building", "vehicle", "gravel", "person", "other"]}
#    colordict_model = {"water": [255, 0, 0], "roadpavedarea": [80, 80, 80], "grass" : [0, 255, 0], "tree" : [0, 127, 0], "rooftop" : [127, 0, 127], "other": [255, 255, 255]}


#output = {"tree" : ["tree"], "grass" : ["grass"], "roadpavedarea" : ["road", "pavedarea"],
#          "other" : ["building", "vehicle", "water", "gravel", "person", "rooftop", "other"]}
#output = {"tree" : ["tree"], "grass" : ["grass"], "roadpavedarea" : ["road", "pavedarea"], "water" : ["water"],
#          "other" : ["building", "vehicle", "gravel", "person", "rooftop", "other"]}
#output = {"tree" : ["tree"],
#          "other" : ["building", "vehicle", "water", "gravel", "person", "road", "pavedarea", "rooftop", "grass", "other"]}
#output = {"water" : ["water"], "roadpavedarea" : ["road", "pavedarea"],
#          "other" : ["building", "vehicle", "gravel", "person", "tree", "rooftop", "grass", "other"]}
#output = {"tree" : ["tree"], "roadpavedarea" : ["road", "pavedarea"],
#          "other" : ["building", "vehicle", "gravel", "person", "water", "rooftop", "grass", "other"]}

#outputcategories = ["building", "vehicle", "road", "gravel", "water", "person", "tree", "pavedarea", "rooftop", "grass",                     "other"]


categories = list(output.keys())
print(categories)
#colordict_model = {"tree" : [0, 127, 0], "roadpavedarea" : [80, 80, 80], "grass" : [0, 255, 0], "other" : [255, 255, 255]}
#colordict_model = {"tree" : [0, 127, 0], "roadpavedarea" : [80, 80, 80], "grass" : [0, 255, 0], "water" : [255, 0, 0], "other" : [255, 255, 255]}
#colordict_model = {"tree" : [0, 127, 0], "other" : [255, 255, 255]}
#colordict_model = {"water" : [0, 0, 0], "roadpavedarea" : [80, 80, 80], "other" : [255, 255, 255]}
#colordict_model = {"tree" : [0, 127, 0], "roadpavedarea" : [80, 80, 80], "other" : [255, 255, 255]}

colordict_original = {"building": [0, 0, 255], "vehicle": [0, 127, 255], "road": [80, 80, 80],
                  "gravel": [255, 127, 255],
                  "water": [255, 0, 0], \
                  "person": [0, 255, 255], "tree": [0, 127, 0], "pavedarea": [220, 220, 220],
                  "rooftop": [127, 0, 127], "grass": [0, 255, 0], \
                  "other": [0, 0, 0]}


outputdict = {}
for k in range(len(categories)):
    print(k)
    for c in output[categories[k]]:
        print(c)
        outputdict[c] = colordict_model[categories[k]]

print(outputdict)

if IoU:
    performanceIoU()


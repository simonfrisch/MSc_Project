#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:50:31 2020
This is a script that automatically creates a dataset of satellite pictures
and their corresponding masks (at the moment water/non-water). The dataset
will be used to train a CNN.
@author: alexschindler
"""

import requests
import cv2 as cv
import pandas as pd
import os
import random

os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Python\\GeoPy\\Data\\Train")

# get method of requests module
# return response object


def getpicture(htmltag):  # function gets the pictures from the Google API
    filename = "temp.png"
    r = requests.get(htmltag)  # get the file from Google API
    f = open(filename, 'wb')  # save and re-open the picture - is there a better way?
    f.write(r.content)  # r.content returns a byte object
    f.close()
    return (cv.imread(filename))


def makewatermask(picturecounter, croppedmap):  # saves the watermask and returns the nr of waterpixels
    waterpixels = 0  # counts number of waterpixels
    for p in range(width):  # go through the pixels and create masks
        for q in range(length):
            if ((croppedmap[p, q, :] == watercolor).all()):  # == water pixels
                croppedmap[p, q, :] = [0, 0, 0]  # make water black
                waterpixels += 1
            else:
                croppedmap[p, q, :] = [255, 255, 255]
    mappicturename = "Masks/" + str(picturecounter) + "_mask.png"
    cv.imwrite(mappicturename, croppedmap)
    return (waterpixels)


def makesatellitepic(picturecounter, croppedsatellite):  # saves the satellite picture
    satellitepicturename = "Satellite/" + str(picturecounter) + "_satellite.png"
    cv.imwrite(satellitepicturename, croppedsatellite)
    return


def makepictures(latitude, longitude, zoom, startpoint):  # this function makes the pics (satellite + masks)
    global picturecounter  # declare the global variable that counts pictures
    print("working on picture " + str(picturecounter))

    url = "https://maps.googleapis.com/maps/api/staticmap?"
    #api_key = "AIzaSyDYW3ZfnG9T6gaVohQ4JTJPv52l_5x7bLk" # api key from alex
    api_key = "AIzaSyAQE6QD5tsoHKNmu1TrTO8IZcXK2jjjdCo" # my api key
    style = "&style=element:labels%7Cvisibility:off&style=feature:administrative%7Celement:geometry%7Cvisibility:off&style=feature:administrative.land_parcel%7Cvisibility:off&style=feature:administrative.neighborhood%7Cvisibility:off&style=feature:poi%7Cvisibility:off&style=feature:road%7Celement:labels.icon%7Cvisibility:off&style=feature:transit%7Cvisibility:off"

    htmlmap = url + "center=" + str(latitude) + "," + str(longitude) + \
              "&zoom=" + str(zoom) + "&size=" + str(length) + "x" + str(width + 30) + \
              style + "&maptype=roadmap&key=" + api_key

    htmlsatellite = url + "center=" + str(latitude) + "," + str(longitude) + \
                    "&zoom=" + str(zoom) + "&size=" + str(length) + "x" + str(width + 30) + \
                    style + "&maptype=satellite&key=" + api_key

    mappicture = getpicture(htmlmap)
    satellitepicture = getpicture(htmlsatellite)
    picturecounter += 1

    mapcropped = mappicture[0:width, :]
    watercount = makewatermask(picturecounter, mapcropped)  # make and save the watermask

    satellitecropped = satellitepicture[0:width, :]
    makesatellitepic(picturecounter, satellitecropped)

    picturedata.loc[int(picturecounter)] = [picturecounter] + [startpoint[0]] + \
                                           [latitude] + [longitude] + [startpoint[1]] + [startpoint[2]] + [int(zoom)] \
                                           + [100 * watercount / (length * width)]  # the new row in picture data table

    print(picturedata.loc[int(picturecounter)])


def makepicturesetrandom(startpoints, nrofpictures,
                         zoomlevels):  # this takes pictures and changes the coordinates by using random noise
    nrofstartpoints = len(startpoints.keys())  # nr of startpoints
    for i in range(nrofstartpoints):
        startpoint = startpoints[i]

        for zoom in zoomlevels:
            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            makepictures(latitude, longitude, zoom, startpoint)  # take the first picture at the startpoint

            for k in range(nrofpictures - 1):
                random.seed(123)
                noise_param = 0.005
                rand_long = random.uniform(-noise_param,
                                           noise_param)  # generate a small uniform random number around zero
                rand_lat = random.uniform(-noise_param,
                                          noise_param)  # generate a small uniform random number around zero

                latitude += rand_lat  # add the random number to the latitude
                longitude += rand_long  # add the random number to the longitude
                makepictures(latitude, longitude, zoom, startpoint)


def makepicturesetstar(startpoints, nrofpictures,
                       zoomlevels):  # this takes pictures starting in the middle and going in 8 directions (like a star)
    nrofstartpoints = len(startpoints.keys())  # nr of startpoints
    for i in range(nrofstartpoints):
        startpoint = startpoints[i]

        for zoom in zoomlevels:
            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            makepictures(latitude, longitude, zoom, startpoint)

            for k in range((nrofpictures - 1) // 8):
                latitude += 0.002  # going 'up' or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                latitude -= 0.002  # going 'down' or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                longitude += 0.002  # going 'right' or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                longitude -= 0.002  # going 'left' or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                latitude += 0.002  # or whatever we want to take
                longitude += 0.002  # or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                latitude -= 0.002  # or whatever we want to take
                longitude -= 0.002  # or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                latitude += 0.002  # or whatever we want to take
                longitude -= 0.002  # or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)

            latitude = startpoint[1]  # coordinates of the startpoint
            longitude = startpoint[2]
            for k in range(nrofpictures // 8):
                latitude -= 0.002  # or whatever we want to take
                longitude += 0.002  # or whatever we want to take
                makepictures(latitude, longitude, zoom, startpoint)


picturedata = pd.DataFrame(
    columns=['Nr', 'Place', 'Longitude', 'Latitude', 'Start longitude', 'Start latitude', 'Zoom', 'Percent water'])

picturecounter = 0  # global variable that counts pictures

length = 640  # length and width of pictures
width = 480
watercolor = [255, 218, 170]  # color of water

# please change the path
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Python\\GeoPy\\Data\\Train")
# os.chdir("/content/images")

startpointstrain = {0: ("Zurich", 47.373462, 8.542265), 1: ("Delhi", 28.621642, 77.254400),
                    2: ("New York", 40.706001, -73.996038)}  # coordinates of the startpoints

zoomlevelstrain = [17]  # defines the zoom level of the map - could also be a list of zoomlevels

nrofpicturestrain = 30  # the total number of pictures we want per start point

makepicturesetrandom(startpointstrain, nrofpicturestrain, zoomlevelstrain)

print(picturedata)
picturedata.to_excel("picturedata_train.xlsx", index=False)  # save the information as an Excel table

#os.remove("temp.png")  # removes the temporary picture

os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Python\\GeoPy\\Data\\Test")
# os.chdir("/content/images")

startpointstest = {0: ("Lucerne", 47.053348, 8.300113),
                   1: ("Moscow", 55.744470, 37.537896)}  # coordinates of the startpoints

zoomlevelstest = [17]  # defines the zoom level of the map - could also be a list of zoomlevels

nrofpicturestest = 30  # the total number of pictures we want per start point

makepicturesetrandom(startpointstrain, nrofpicturestrain, zoomlevelstrain)

print(picturedata)
picturedata.to_excel("picturedata_test.xlsx", index=False)  # save the information as an Excel table

#os.remove("temp.png")  # removes the temporary picture

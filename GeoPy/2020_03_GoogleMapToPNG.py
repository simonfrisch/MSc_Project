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
  
# get method of requests module 
# return response object 


def getpicture(htmltag): # function gets the pictures from the Google API
    filename = "temp.png"
    r = requests.get(htmltag) # get the file from Google API
    f = open(filename, 'wb') # save and re-open the picture - is there a better way?
    f.write(r.content) # r.content returns a byte object
    f.close()
    return(cv.imread(filename))

def makewatermask(picturecounter, croppedmap): # saves the watermask and returns the nr of waterpixels
    waterpixels = 0 # counts number of waterpixels
    for p in range(width): # go through the pixels and create masks
        for q in range(length):
            if((croppedmap[p, q, :] == watercolor).all()): # == water pixels
                croppedmap[p, q, :] = [0, 0, 0] # make water black
                waterpixels += 1
            else:
                croppedmap[p, q, :] = [255, 255, 255]
    mappicturename = "Masks/" + str(picturecounter) + "_mask.png"
    cv.imwrite(mappicturename, croppedmap)
    return(waterpixels)

def makesatellitepic(picturecounter, croppedsatellite): #saves the satellite picture
    satellitepicturename = "Satellite/" + str(picturecounter) + "_satellite.png"
    cv.imwrite(satellitepicturename, croppedsatellite)
    return


def makepictures(latitude, longitude, zoom): # this function makes the pics (satellite + masks)
    global picturecounter # declare the global variable that counts pictures
    print("working on picture " + str(picturecounter))
    
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    api_key = "AIzaSyDYW3ZfnG9T6gaVohQ4JTJPv52l_5x7bLk"
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
    
    mapcropped = mappicture[0:width,:]
    watercount = makewatermask(picturecounter, mapcropped) # make and save the watermask
    
    satellitecropped = satellitepicture[0:width,:]
    makesatellitepic(picturecounter, satellitecropped)

    picturedata.loc[int(picturecounter)] = [picturecounter] + \
    [latitude] + [longitude] + [int(zoom)] \
    + [100 * watercount / (length * width)] # the new row in picture data table
    
picturedata = pd.DataFrame(columns = ['Nr','Longitude', 'Latitude', 'Zoom', 'percent water'])

picturecounter = 0 # global variable that counts pictures

length = 640 # length and width of pictures
width = 480
watercolor = [255, 218, 170] # color of water


startpoints = {0 : (47.373462, 8.542265), 1 : (28.621642, 77.254400), 2 : (40.706001, -73.996038)} # coordinates of the startpoints

zoom = 17 # defines the zoom level of the map - could also be a list of zoomlevels

nrofpictures = 25 # the total number of pictures we want per location
nrofstartpoints = len(startpoints.keys()) # nr of startpoints

# please change the path
os.chdir("D:\\Documents\\UZH\\FS20\\MSc Project\\Python\\GeoPy\\Data")

for p in range(nrofstartpoints):
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    makepictures(latitude, longitude, zoom)
    
    for _ in range(nrofpictures // 8):
        latitude += 0.002 # going 'up' or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        latitude -= 0.002 # going 'down' or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        longitude += 0.002 # going 'right' or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        longitude -= 0.002 # going 'left' or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        latitude += 0.002 # or whatever we want to take
        longitude += 0.002 # or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        latitude -= 0.002 # or whatever we want to take
        longitude -= 0.002 # or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        latitude += 0.002 # or whatever we want to take
        longitude -= 0.002 # or whatever we want to take
        makepictures(latitude, longitude, zoom)
    
    latitude = startpoints[p][0] # coordinates of the startpoint
    longitude = startpoints[p][1]
    for _ in range(nrofpictures // 8):
        latitude -= 0.002 # or whatever we want to take
        longitude += 0.002 # or whatever we want to take
        makepictures(latitude, longitude, zoom)

print(picturedata)
picturedata.to_excel("picturedata.xlsx", index=False) # save the information as an Excel table

os.remove("temp.png") # removes the temporary picture
        


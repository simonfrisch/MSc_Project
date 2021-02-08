"""
This script adds random noise to a given position (longitude and latitude)

@author: Simon Frischknecht
"""

import random
import warnings
warnings.filterwarnings("ignore")
from geopy.geocoders import Nominatim
geolocator = Nominatim()

long = 47.315978 # initial longitude
lat = 8.2085177 # initial latitude

random.seed(123)
noise_param = 0.005
rand_long = random.uniform(-noise_param,noise_param) # generate a small uniform random number around zero
rand_lat = random.uniform(-noise_param,noise_param) # generate a small uniform random number around zero

long_new = str(long + rand_long) # add the random number to the longitude
lat_new = str(lat + rand_lat) # add the random number to the latitude
long_lat = long_new + "," + lat_new # concatenate the coordinates
print(long_new, lat_new)

# create link for direct access to google maps:
print("https://www.google.ch/maps/@" + long_new + "," + lat_new + ",1804m/data=!3m1!1e3")
location = geolocator.reverse(long_lat)
print(location.address)

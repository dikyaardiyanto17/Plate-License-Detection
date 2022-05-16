import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

#Notes : I separated the process for 4 steps for to make easier to understand the code, you should combine the 4 steps to try this code

################################
count = "1"
#I use this variable to make my data more convenient, so you can use it or simply remove the variable

image = cv.imread(r'C:\Users\dikya\PycharmProjects\pythonProject\pythonProject\OPENCV2\venv\Data\Motor\D ('+count+').jpg')
#Locate plate license image, in this case use photographed image using a phone, you can take a real time image uisng video capture or something similiar
#I will try to upload the realtime image capture in different code
#I'm using photographed image because of my device is not highspec and take a bit of time

image = cv.resize(image, (int(image.shape[1] * .4), int(image.shape[0] * .4)))
#Resizing the image

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#Convert to gray image

matriks = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
#Make a cernel

image1 = cv.morphologyEx(image_gray, cv.MORPH_OPEN, matriks)
image_normal = image_gray - image1
(thresh, image_normalisasi_bw) = cv.threshold(image_normal, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
(thresh, image_tanpa_normalisasi_bw) = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#Normalize the image
#You can make the result appear using cv.imshow('variabel','name variabel')
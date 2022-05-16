import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

contour_kendaraan, hierarchy = cv.findContours(image_normalisasi_bw, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)
#Use contour to locate the image, you can search for yourself what is contour and how its work if you wanted to understand it better

indek_kandidat_plat = []
indek_counter_kontor_kendaraan = 0
for contour_Ken in contour_kendaraan:
    x, y, w, h = cv.boundingRect(contour_Ken)
    aspect_ratio = w / h
    if w >= 200 and aspect_ratio <= 4:
        indek_kandidat_plat.append(indek_counter_kontor_kendaraan)

    indek_counter_kontor_kendaraan += 1
#I'm using this setting for motorcycle plat license

image_tunjukan_plat = image.copy()
image_tunjukan_plat_bw = cv.cvtColor(image_normalisasi_bw, cv.COLOR_GRAY2RGB)

if len(indek_kandidat_plat) == 0:
    print("Plat nomor tidak ditemukan")
elif len(indek_kandidat_plat) == 1:
    x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contour_kendaraan[indek_kandidat_plat[0]])
    cv.rectangle(image_tunjukan_plat, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    cv.rectangle(image_tunjukan_plat_bw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    image_plat_nomor_gray = image_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]
else:
    print('Dapat dua lokasi plat, pilih lokasi plat kedua')
    x_plate, y_plate, w_plate, h_plate = cv.boundingRect(contour_kendaraan[indek_kandidat_plat[1]])
    cv.rectangle(image_tunjukan_plat, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    cv.rectangle(image_tunjukan_plat_bw, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    image_plat_nomor_gray = image_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

#In this process you could get the plat license and maybe two candidate, in this code i made the second candidate because its close to the shape of plate license
#Dont mind the complicated looks in this, those code are for drawing the line around the plate license with some distance (Refer to matriks variable)
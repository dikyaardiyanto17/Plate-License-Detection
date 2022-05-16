import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

    image_tinggi = 40
    image_lebar = 40
#Set the width and length of dataset

    klasifikasi_nama = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#Classify the dataset

    model = keras.models.load_model(r'C:\Users\dikya\PycharmProjects\pythonProject\pythonProject\OPENCV2\venv\Reading 2')
#Locate the dataset

    plat_nomor = []

    for karakter_terurut in indek_karakter_terurut:
        x, y, w, h = cv.boundingRect(contours_plat_nomor[karakter_terurut])

        potongan_karakter = cv.cvtColor(image_plat_bw[y:y + h, x:x + w], cv.COLOR_GRAY2BGR)

        potongan_karakter = cv.resize(potongan_karakter, (image_lebar, image_tinggi))

        array_image = keras.preprocessing.image.img_to_array(potongan_karakter)

        array_image = tf.expand_dims(array_image, 0)

        prediksi = model.predict(array_image)
        skor = tf.nn.softmax(prediksi[0])

        plat_nomor.append(klasifikasi_nama[np.argmax(skor)])
        print(klasifikasi_nama[np.argmax(skor)], end='')

    nomor_plat = ''
    for a in plat_nomor:
        nomor_plat += a
#Those code will apply the dataset to image to make the character readable

    cv.putText(image_tunjukan_plat, nomor_plat, (x_plate, y_plate + h_plate + 50), cv.FONT_ITALIC, 2.0, (0, 255, 0), 3)
    cv.imwrite(r'C:\Users\dikya\PycharmProjects\pythonProject\pythonProject\OPENCV2\venv\ScanFolder\S'+nomor_plat+'.jpg', image_tunjukan_plat)
    cv.imshow(nomor_plat, image_tunjukan_plat)
#and this is to write the plate license number on the image


cv.waitKey(0)

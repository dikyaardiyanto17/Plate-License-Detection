import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras


#Same as before, i'm using contour to locate the characters
(thresh, image_plat_bw) = cv.threshold(image_plat_nomor_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

matriks = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

image_plat_bw = cv.morphologyEx(image_plat_bw, cv.MORPH_OPEN, matriks)

contours_plat_nomor, hierarchy = cv.findContours(image_plat_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

indek_kandidaat_karakter = []

indek_counter_contour_plat_nomor = 0

image_plat_nomor_rgb = cv.cvtColor(image_plat_nomor_gray, cv.COLOR_GRAY2BGR)
image_plat_nomor_bw_rgb = cv.cvtColor(image_plat_bw, cv.COLOR_GRAY2RGB)
#cv.imwrite(r'C:\Users\dikya\PycharmProjects\pythonProject\pythonProject\OPENCV2\venv\ScanFolder\Kandidat\S'+ count +'.jpg', image_plat_nomor_bw_rgb)
#Dont mind this, its just some of my trial and error for research purpose

for contour_plat in contours_plat_nomor:

    x_char, y_char, w_char, h_char = cv.boundingRect(contour_plat)

    if h_char >= 40 and h_char <= 60 and w_char >= 10:
        indek_kandidaat_karakter.append(indek_counter_contour_plat_nomor)

        cv.rectangle(image_plat_nomor_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)
        cv.rectangle(image_plat_nomor_bw_rgb, (x_char, y_char), (x_char + w_char, y_char + h_char), (0, 255, 0), 5)

    indek_counter_contour_plat_nomor += 1
#Finding the character, and set some condition so expired date of plate license wont get detected

if indek_kandidaat_karakter == []:

    print('Tidak Menemukan Karakter')
else:

    skor_kandidat_karakter = np.zeros(len(indek_kandidaat_karakter))

    counter_indek_kandidat_karakter = 0

    for kandidat_karakter_A in indek_kandidaat_karakter:


        xA, yA, wA, hA = cv.boundingRect(contours_plat_nomor[kandidat_karakter_A])
        for kandidat_karakter_B in indek_kandidaat_karakter:

            if kandidat_karakter_A == kandidat_karakter_B:
                continue
            else:
                xB, yB, wB, hB = cv.boundingRect(contours_plat_nomor[kandidat_karakter_B])

                y_difference = abs(yA - yB)

                if y_difference < 11:
                    skor_kandidat_karakter[counter_indek_kandidat_karakter] = skor_kandidat_karakter[
                                                                               counter_indek_kandidat_karakter] + 1

        counter_indek_kandidat_karakter += 1

    print(skor_kandidat_karakter)

    karakter_indek = []

    counter_karakter = 0

    for skor in skor_kandidat_karakter:
        if skor == max(skor_kandidat_karakter):
            karakter_indek.append(indek_kandidaat_karakter[counter_karakter])
        counter_karakter += 1
#Those code are for compare the finding character with previous finding character and if the difference reached more than what i set, its will get eliminated from character candidates

    image_plat_nomor_rgb2 = cv.cvtColor(image_plat_nomor_gray, cv.COLOR_GRAY2BGR)

    for karakter in karakter_indek:
        x, y, w, h = cv.boundingRect(contours_plat_nomor[karakter])
        cv.rectangle(image_plat_nomor_rgb2, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv.putText(image_plat_nomor_rgb2, str(karakter_indek.index(karakter)), (x, y + h + 50), cv.FONT_ITALIC, 2.0, (0, 0, 255), 3)

    koordinat_x = []

    for karakter in karakter_indek:
        x, y, w, h = cv.boundingRect(contours_plat_nomor[karakter])

        koordinat_x.append(x)

    koordinat_x = sorted(koordinat_x)

    indek_karakter_terurut = []

    for koor_x in koordinat_x:
        for karakter in karakter_indek:

            x, y, w, h = cv.boundingRect(contours_plat_nomor[karakter])

            if koordinat_x[koordinat_x.index(koor_x)] == x:
                indek_karakter_terurut.append(karakter)

    image_plat_nomor_rgb3 = cv.cvtColor(image_plat_nomor_gray, cv.COLOR_GRAY2BGR)
#Sorted the character from left to right

    for karakter_terurut in indek_karakter_terurut:
        x, y, w, h = cv.boundingRect(contours_plat_nomor[karakter_terurut])

        cv.rectangle(image_plat_nomor_rgb3, (x, y), (x + w, y + h), (0, 255, 0), 5)

        cv.putText(image_plat_nomor_rgb3, str(indek_karakter_terurut.index(karakter_terurut)), (x, y + h + 50), cv.FONT_ITALIC, 2.0,
                   (0, 0, 255), 3)
#Those code are just for make a line like before
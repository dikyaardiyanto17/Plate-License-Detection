import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#https://www.tensorflow.org/tutorials/images/classification

data_dir = r'C:\Users\dikya\PycharmProjects\pythonProject\pythonProject\OPENCV2\venv\dataset'
#Locate the dataset

ukuran_batch = 32
tinggi_image = 40
lebar_image = 40
#Set dataset image

pelatihan_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=550,
  image_size=(tinggi_image, lebar_image),
  batch_size=ukuran_batch)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=550,
  image_size=(tinggi_image, lebar_image),
  batch_size=ukuran_batch)

#Set training and validation ratio

class_names = pelatihan_dataset.class_names

AUTOTUNE = tf.data.AUTOTUNE
#Made the dataset readable for progam

pelatihan_dataset = pelatihan_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

normaliasasi_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalisasi_dataset = pelatihan_dataset.map(lambda x, y: (normaliasasi_layer(x), y))
gambar_batch, label_batch = next(iter(normalisasi_dataset))
Gambar = gambar_batch[0]
#Normalization for dataset

klasifikasi = 36
#Classify the dataset with number of your classification, in which case 36 character (A-Z, 0-9)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(tinggi_image, lebar_image, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(klasifikasi)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#You modify convolution proccess for better result

model.summary()

epochs=25
history = model.fit(
  pelatihan_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)
#Modify epoch to make a model with a better accuracy

akurasi = history.history['accuracy']
validasi_akurasi = history.history['val_accuracy']

hilang = history.history['loss']
validasi_hilang = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, akurasi, label='Training Accuracy')
plt.plot(epochs_range, validasi_akurasi, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, hilang, label='Training Loss')
plt.plot(epochs_range, validasi_hilang, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('Last Model5')

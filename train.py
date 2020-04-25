"""
Grupo de Sistemas en Tiempo Real

Autor: Ing. Gonzalez Lucas Ezequiel

Script para realizar Transfer Learning sobre una arquitectura
pre-entrenada ResNet50
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import warnings
warnings.filterwarnings('ignore')

# path donde se alojan la base de datos de imagenes
# Dentro del path tenemos 3 carpetas : Train, Validacion, Test
# Dentro de las anteriores tenemos carpetas para imagenes con presencia de incedio
# y con imagenes sin incendios
input_path = "/content/drive/My Drive/Colab Notebooks/DataBase/"

train_datagen = ImageDataGenerator(
    shear_range = 10,
    zoom_range = 0.2,
    horizontal_flip = True,
    preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(
    input_path + 'train',
    batch_size = 32,
    class_mode = 'binary',
    target_size = (224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    input_path + 'valid',
    shuffle = False,
    class_mode = 'binary',
    target_size = (224,224))

conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)

optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch = 700 // 32, 
                              epochs = 20,
                              validation_data = validation_generator,
                              validation_steps = 1000  
                             )
            

# save
model.save('/content/drive/My Drive/Colab Notebooks/modelos/model_3.h5')
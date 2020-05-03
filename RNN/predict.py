"""
##########################################
  Autor : Ing.Gonzalez Lucas Ezequiel
  Fecha : 10/12/2019
  Universidad Nacional de Rio Cuarto
  Grupo de sistemas en tiempo real (GSTR)
##########################################
"""

import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import numpy as np

# load model
path_rasp = '/home/ubuntu/proyecto2019/resnet50/'
path_local = '/content/drive/My Drive/Colab Notebooks/modelos/model.h5'
model = load_model(path_rasp + 'model.h5')

def Predict(imagen):
  # Esta funcion realiza la prediccion sobre la imagen de entrada
  # input :
  #       - imagen : ruta absolupta de la image donde se aloja
  # output: 
  #       - answer : lista que contiene las probabilidades con que la red
  #         estima que sea una imagen con incendio o no [p_incendio, p_no_incendio 
  x = load_img(imagen, target_size=(224, 224))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  # Prediccion
  answer = model.predict(x).tolist()
  return answer


imagen = 'img.jpg'
#print(Predict(path_rasp + imagen))
resultado = Predict(path_rasp + imagen)
#print(resultado)
print('Probabilidad de incendio: %'.format(np.around(resultado[0][0],2)))
print('Probabilidad de no incendio: %'.format(np.around(resultado[0][1],2)))

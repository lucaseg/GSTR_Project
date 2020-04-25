"""
##########################################
  Autor : Ing.Gonzalez Lucas Ezequiel
  Fecha : 10/12/2019
  Universidad Nacional de Rio Cuarto
  Grupo de sistemas en tiempo real (GSTR)
##########################################
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

import os
from keras.preprocessing.image import load_img, img_to_array
# Directorios de Test
path_ni = '/content/drive/My Drive/Colab Notebooks/DataBase/no_incendio/test'
path_i = '/content/drive/My Drive/Colab Notebooks/DataBase/incendio/test'
# load
model = load_model('/content/drive/My Drive/Colab Notebooks/modelos/model.h5')

def TestNetwork(path):
  test = list()
  list_dir_org = os.walk(path)
  for root, dirs, files in list_dir_org:
    for imagen in files:
      if imagen != '.DS_Store':
        x = load_img(path + '/' + imagen, target_size=(224, 224))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        answer = model.predict(x).tolist()
        test.append(answer)
      else:
        pass
  # Primera componente de la lista probabilidad de incendio
  # Segunda componente probabilidad no incendio
      
  return test

test_i = TestNetwork(path_i)
test_ni = TestNetwork(path_ni)

def ResultadosTest(probabilidades):
  # Primera componente de la lista probabilidad de incendio
  # Segunda componente probabilidad no incendio
  componente_1 = 0
  componente_2 = 0
  for vector in probabilidades:
    if vector[0][0] > vector[0][1]:
      componente_1 += 1
    else:
      componente_2 +=1
  return [(componente_1 / len(probabilidades))*100, (componente_2 / len(probabilidades))*100]

eval_test_incendio = ResultadosTest(test_i)
eval_test_noincendo = ResultadosTest(test_ni)

print("Test sobre imagenes de incendio")
print("Predicciones correctas: {}% , Predicciones incorrectas: {}%".format(round(eval_test_incendio[0],2), round(eval_test_incendio[1], 2)))

print("Test sobre imagenes de sin incendio")
print("Predicciones correctas: {}% , Predicciones incorrectas: {}%".format(round(eval_test_noincendo[1], 2), round(eval_test_noincendo[0],2)))
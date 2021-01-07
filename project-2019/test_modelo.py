import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import time
import os

nombre = []

incendio = 0
no_incendio = 0

dir_incendio = "./data/test/incendio"
dir_noincendio = "./data/test/no_incendio"

def image_list(directorio):
  global nombre
  for item in os.listdir(directorio):
      (nombreFichero, extension) = os.path.splitext(item)
      if(extension == ".jpg" or extension == ".jpeg" or extension == ".png"):
          nombre.append(item)
        
def predict(file,directorio):
  global incendio
  global no_incendio
  direccion_file = directorio+file
  x = load_img(direccion_file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    incendio += 1
    
  elif answer == 1:
    no_incendio += 1
    

image_list(dir_incendio)    
for imagen in nombre:
  predict(imagen,"./data/test/incendio/")
  
print ("El numero de incendio: ",incendio)
print ("El numero de No_incendio: ",no_incendio)

nombre = []
incendio = 0
no_incendio = 0

image_list(dir_noincendio)
for imagen in nombre:
  predict(imagen,"./data/test/no_incendio/")
  
print ("El numero de incendio: ",incendio)
print ("El numero de No_incendio: ",no_incendio)  
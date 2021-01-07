# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from scipy import ndimage
from time import time
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Script beta para la deteccion temprana de incendios mediante TDI
# Con el uso de redes neuronales la cual actua como clasificador 
# este ultimo es quien decide si en la imagen tomada existe algun tipo de incendio
# en caso de ser afirmativo se procede a realizar un tratamiento a la imagen 
# para filtrar la zona afectada
# este tratamiento se basa en ...


longitud, altura = 450, 450
modelo = './modelo_4/modelo.h5'
pesos_modelo = './modelo_4/pesos.h5'

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  cnn = load_model('./modelo_4/modelo.h5')

cnn.load_weights(pesos_modelo)

# Los argumentos de entrada para ejecutar el script son:
# 1) file_name que hace referencia al nombre de la imagen a procesar
# 2) ro, que es el factor para el tratamiento digital de la imagen
start_time = time()
file_name = img1.jpg
ro = 1

######################## Declaracion de funciones ######################
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Se detecto incendio")
  elif answer == 1:
    print("No se detecto incendio")

def graficador (imagen,titulo):
    # La funcion realiza graficos con matplotlib
    # imagen [str] : nombre de la imagen a graficar
    # titulo [str]: Titulo del grafico
    figure()
    plt.imshow(imagen)
    plt.xticks([]), plt.yticks([])
    plt.title(titulo)
    plt.savefig(titulo +'.jpg')


def arrayImage(array,factor,nombre,save):
    # La funcion transforma un array en Imagen y la guarda a eleccion
    # array  : es el array a transformar
    # factor[float,Int] : factor de escalamiento
    # nombre [str] : nombre para guardar la imagen con extension
    # save [True-False] : para dar la opcion de guardar o no la imagen
    array = around(array*factor)
    imagen = Image.fromarray(array)
    imagen_gray = imagen.convert('L')
    if save : 
        imagen_gray.save(nombre)
    else : 
        pass

def frange(start,stop,step):
    # Esta funcion la utilice para poder hacer un range aplicado en un for
    # que nos permita ir con pasos racionales (no enteros)
    i = start
    while i < stop:
        yield i
        i += step

def histograma_RGB():
    # Grafico del Histograma de la imagen en sus 3 canales RGB
    figure()
    subplot(2,3,4)
    hist(array_r.flatten(),bins = 256)
    title("histograma red")
    subplot(2,3,5)
    hist(array_g.flatten(),bins = 256)
    title("histograma green")
    subplot(2,3,6)
    hist(array_b.flatten(),bins = 256)
    title('histograma blue')
    subplot(2,3,2)
    imshow(im)
    title('original')
    plt.savefig('Histograma.jpg')

def dibujar_contornos(original,umbralizada,numero):
    # Esta funcion nos permite realizar un trazado del contorno detectado en la segmentacion
    # dibujada sobre la imagen original 
    dst = umbralizada
    src = cv2.imread(str(original))
    contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite('contornos'+str(numero)+'.jpg', src)

########################### Fin Declaracion de funciones############################

def main(ro,file_name):
    i = 1
    # ro: Indice de ponderacion
    # file : Archivo imagen a procesar
    nombre = file_name
    
    # Apertura de la imagen a tratar
    im = Image.open(str(file_name))

    # Separamos la imagen original en sus 3 canales RGB
    r,g,b=im.split()

    # Convertimos los canales RGB a Array para poder operar entre ellos
    array_r = np.array(r)
    array_g = np.array(g)
    array_b = np.array(b)

    # Normalizacion a valores entre 0 y 1
    r = array_r/255.0 
    g = array_g/255.0 
    b = array_b/255.0 

    # Para ver resultados despues borrar
    r_g = r - g
    r_b = r - b
    
    # Indice de filtrado
    FDI = 2*r - g - b
    FFDI= (r*(2*ro + 1)) - (g*(ro + 2)) + (b*(1 - ro))

    # Guardo la imagen resultante de la FDI
    arrayImage(FDI,255,'FDI'+str(ro)+'.jpg',True)
    # Guardo la imagen resultante de FFDI
    arrayImage(FFDI,255,'FFDI'+str(ro)+'.jpg',True)
    #graficador(FFDI,'FFDI_grafico'+str(ro))

    # Aqui se busca eliminar ruido del resultado final
    dim = FFDI.shape         # Dimensiones de la matriz
    array = around(FFDI*255) # Redondeo de valores
    imagen = Image.fromarray(array)   # Conversion de array a imagen
    imagen_gray = imagen.convert('L') # Conversion a escala de grises
    med_denoised = ndimage.median_filter(imagen_gray, size = 20)  # Filtrado por mediana
    cv2.imwrite('FFDI_filtrada'+str(ro)+'.jpg',med_denoised)      # Guardado de resultado

    # Umbralizacion adaptable utilizando el metodo otsu
    res_ffdi = cv2.imread('FFDI_filtrada'+str(ro)+'.jpg',0)
    blur = cv2.GaussianBlur(res_ffdi,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    dibujar_contornos(nombre,th3,i)
    cv2.imwrite('FFDI_umbralizada'+str(ro)+'.jpg', th3)

    show()

if __name__ == "__main__":
    predict(file_name)
    main(ro,file_name)
    elapse_time = time() - start_time
    print("Tiempo de ejecucion: ",elapse_time)

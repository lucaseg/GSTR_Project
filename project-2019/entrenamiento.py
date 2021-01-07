import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()


# Directorios donde se encuentran las imagenes de entrenamiento y validacion
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Parameters
"""
# epocas son las veces en que se va a iterar
epocas=10
# Tamaño de las imagenes para procesar originalmente usamos 450 x 450
longitud, altura = 150, 150
# Es el numero de imagenes que le daremos a la computadora a procesar en cada uno de los pasos
batch_size = 32
# Es el numero de veces que se procesa la informacion en cada epoca
pasos = 500
# Al final de cada epoca se corra con el set de datos de validacion y asi ver que tan bien esta aprendiendo
validation_steps = 100
# El numero de filtros que aplicaremos en cada convolucion
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
# Tamaño del filtro que usaremos en el maxpooling
tamano_pool = (2, 2)
# Cantidad de clases que tenemos en el programa
clases = 2
# Que tan grande seran los ajustes para alcanzar un aprendizaje optimo
lr = 0.0005


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #los valores de los pixeles son de 0-1
    shear_range=0.3,  # Realiza una rotacion de 0.2
    zoom_range=0.3,   # Hace un zoom sobre la imagen
    horizontal_flip=True) #Tomara la imagen y la invierte

#Para la validacion queremos que la imagen este tal cual
test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

#print "aca va : "
#print entrenamiento_generador.class_indices

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# Creamos la Red Neuronal Convolucional
cnn = Sequential() #tipo de red secuencial

# Capa de entrada
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

#Capas escondidas
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro1, padding ="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro1, padding ="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

#Capa de salida
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

target_dir = './modelo_4/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo_4/modelo.h5')
cnn.save_weights('./modelo_4/pesos.h5')
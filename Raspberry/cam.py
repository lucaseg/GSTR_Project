"""
##########################################
  Autor : Ing.Gonzalez Lucas Ezequiel
  Fecha : 10/12/2019
  Universidad Nacional de Rio Cuarto
  Grupo de sistemas en tiempo real (GSTR)
##########################################
"""

from picamera import PiCamera
from time import sleep

path = '/home/ubuntu/proyecto2019/resnet50/'

camera= PiCamera()

camera.start_preview(alpha=192)
sleep(4)
print('Fotoo')
camera.capture(path + 'img_prueba.jpg')
camera.stop_preview()

# GSTR_Project
Este proyecto fue realizado durante mi segundo año como becario en el grupo de sistemas en tiempo real (GSTR) que tiene como objetivo la deteccion temprana de incendios por medio del monitoreo mediante un drone sobre zonas con alta probabilidad de siniestros que por medio de la toma de imagenes y pos procesamiento de la misma determinar en tiempo real la existencia de algun foco de incendio con el fin de emitir una alerta y de esta forma poder accionar de forma temprana antes de la propagacion incontrolada del siniestro.

La idea de este proyecto es poder brindar una solucion de bajo costo a la problematica anteriormente expuesta, meadiante el uso de redes neuronales convolucionales, este proyecto es una mejora del realizadon durante el año 2018 donde se implemento una arquitectura realizada por mi donde solo contaba con una base de datos escasa pero que de todas formas lanzo buenos resultados. En esta version se trabajo con mayor profundidad utilizando tecnicas de Transfer Learning, mas especificamente se utilizo una red pre-entrenada con una arquitectura de ResNet50.

# Recursos
Para el entrenamiento de la red neuronal se utilizo la plataforma de Google Colab debido a que este proceso requiere de mucho procesamiento de hardware del cual no disponemos pero Google nos brinda la posibilidad de utilizarlos sin ningun costo.

Una vez entrenada la red, fue montada sobre una RaspBerry Pi 4 con 4Gb de memoria RAM, conjuntamente con una camara para realizar pruebas de funcionamiento real.

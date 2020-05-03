<div>
  <h1> Grupo de Sistemas en Tiempo Real</h1> <img src='/assets/gstr.jpg' heigth='150' width='150' />
</div>

# Introduccion
Este proyecto fue realizado durante mi segundo año como becario en el grupo de sistemas en tiempo real (GSTR) que tiene como objetivo la deteccion temprana de incendios por medio del monitoreo mediante un drone sobre zonas con alta probabilidad de siniestros que por medio de la toma de imagenes y pos procesamiento de la misma determinar en tiempo real la existencia de algun foco de incendio con el fin de emitir una alerta y de esta forma poder accionar de forma temprana antes de la propagacion incontrolada del siniestro.

La idea de este proyecto es poder brindar una solucion de bajo costo a la problematica anteriormente expuesta, meadiante el uso de redes neuronales convolucionales, este proyecto es una mejora del realizadon durante el año 2018 donde se implemento una arquitectura realizada por mi donde solo contaba con una base de datos escasa pero que de todas formas lanzo buenos resultados. En esta version se trabajo con mayor profundidad utilizando tecnicas de Transfer Learning, mas especificamente se utilizo una red pre-entrenada con una arquitectura de ResNet50.

# Recursos
<div>
  <img src='/assets/colab.png' heigth='150' width='150' />
  <img src='/assets/raspbetty.png' heigth='150' width='150' />
</div>

Para el entrenamiento de la red neuronal se utilizo la plataforma de Google Colab debido a que este proceso requiere de mucho procesamiento de hardware del cual no disponemos pero Google nos brinda la posibilidad de utilizarlos sin ningun costo.

Una vez entrenada la red, fue montada sobre una RaspBerry Pi 4 con 4Gb de memoria RAM, conjuntamente con una camara para realizar pruebas de funcionamiento real.


# Funcionamiento
Dentro de la carpeta RNN se encuentran los scripts para realizar el train,test y predict de las imagenes.
Nota: Debido a que la base de datos es demaciado grande para el repositorio no la puedo compartir por este medio, sin embargo puede contactarme y con gusto le pasare las imagenes. email: lgonzalez@ing.unrc.edu.ar

El primer paso sera instalar los requerimientos (librerias de python).
Nota: Este proyecto se realizo con python 3.7

Recomiendo trabajar con un entorno virtual, si nunca trabajo con uno a continuacion le muestro como instalarlo e iniciarlo.
```

 1) instalacion
 pip install virtualenv

 2) Creacion del entorno
 virtualenv -p python3 env

 3) Inicializacion del entorno
 source env/bin/activate

```

Una vez teniendo nuestro entorno de trabajo funcionando procedemos a instalar las librerias necesarias.
```

  /RNN
  pip install -r requeriments.txt

```
Una vez lista la instalacion se procede a correr el script de train.
```

  /RNN
  python train.py 

```
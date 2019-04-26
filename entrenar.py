#-*-coding=utf-8-*-
"""algoritmo de entrenamiento para la neurona, debe
   distinguir diferentes tipos de mangos"""
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
#INICIO DE SESIÓN
K.clear_session()
datos_entrenamiento = "./data/entrenamiento"
datos_validacion = "./data/validacion"
#PARÁMETROS
epocas = 19
altura, longitud = 100, 100
batch_size = 32
pasos = 1500
pasos_validacion = 500
filtrosConv1 = 32
filtrosConv2 = 64
tamanio_filtro1 = (3,3)
tamanio_filtro2 = (2,2)
tamanio_pool = (2,2)
clases = 5 #numero de categorías que tengo
lr = 0.0005
#PROCESAMIENTO DE IMÁGENES
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255, #normalización
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True) #todo esto para dar lugar a diferencias entre las imágenes
validacion_datagen = ImageDataGenerator(
    rescale = 1./255)
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    datos_entrenamiento,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical') #buscará datos de entrenamiento y las procesa
class_dictionary = imagen_entrenamiento.class_indices
print class_dictionary #esto para saber qué código le pertenece a la clasificación
imagen_validacion = validacion_datagen.flow_from_directory(
    datos_validacion,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical')
#CREACIÓN DE LA NEURONA
neurona = Sequential()
neurona.add(Convolution2D(
    filtrosConv1,
    tamanio_filtro1,
    padding='same',
    input_shape=(altura,longitud,3),#CAMBIO
    activation='relu'))
neurona.add(MaxPooling2D(pool_size=tamanio_pool))
neurona.add(Convolution2D(
    filtrosConv2,
    tamanio_filtro2,
    padding='same',
    activation='relu'))
neurona.add(MaxPooling2D(pool_size=tamanio_pool))
#INICIO DE LA CLASIFICACIÓN
neurona.add(Flatten())
neurona.add(Dense(256,activation='relu'))
neurona.add(Dropout(0.5)) #activa el 70% de las neuronas en
                      #cada paso, para que la IA tome
                      #caminos más diversos en la resolución
neurona.add(Dense(50,activation='relu'))
neurona.add(Dense(clases,activation='softmax')) #capa de salida, dará probabilidades
neurona.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=lr),
    metrics=['accuracy']) #compilación de la IA, optimizado con Adam
#ENTRENAMIENTO (se viene lo chido)
neurona.fit_generator(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion)
#guardar IA
dir = "./modelo/"
if not os.path.exists(dir):
    os.mkdir(dir)
neurona.save("./modelo/modelo.h5")
neurona.save_weights("./modelo/pesos.h5")

#-*-coding=utf-8-*-
"""algoritmo de entrenamiento para la neurona, debe
   distinguir lo que es un mango y lo que no"""
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
epocas = 20
altura, longitud = 100, 100
batch_size = 32
pasos = 1000
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamanio_filtro1 = (3,3)
tamanio_filtro2 = (2,2)
tamanio_pool = (2,2)
clases = 3 #numero de categorías que tengo
lr = 0.005
#PROCESAMIENTO DE IMÁGENES
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255, #normalización
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True) #todo esto para dar lugar a diferencias entre las imágenes
data_generator = entrenamiento_datagen.flow_from_directory(datos_entrenamiento, target_size=(150,150), batch_size=32, class_mode='sparse')
class_dictionary = data_generator.class_indices
print class_dictionary
validacion_datagen = ImageDataGenerator(
    rescale = 1/255)
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    datos_entrenamiento,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical') #buscará datos de entrenamiento y las procesa
imagen_validacion = validacion_datagen.flow_from_directory(
    datos_validacion,
    target_size = (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical')
#CREACIÓN DE LA NEURONA
cnn = Sequential()
cnn.add(Convolution2D(
    filtrosConv1,
    tamanio_filtro1,
    padding='same',
    input_shape=(altura,longitud,3),
    activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamanio_pool))
cnn.add(Convolution2D(
    filtrosConv2,
    tamanio_filtro2,
    padding='same',
    activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamanio_pool))
#INICIO DE LA CLASIFICACIÓN
cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5)) #activa el 50% de las neuronas en
                      #cada paso, para que la IA tome
                      #caminos más diversos en la resolución
cnn.add(Dense(50,activation='relu')) #capa adicional, más exactitud
cnn.add(Dense(clases,activation='softmax')) #capa de salida, dará probabilidades
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=lr),
    metrics=['accuracy']) #compilación de la IA, optimizado con Adam
#ENTRENAMIENTO (se viene lo chido)
cnn.fit_generator(
    imagen_entrenamiento,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion)
#guardar IA
dir = "./modelo/"
if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save("./modelo/modelo.h5")
cnn.save_weights("./modelo/pesos.h5")
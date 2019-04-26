#-*-coding=utf-8-*-
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    cnn = load_model(modelo)
cnn.load_weights(pesos)
#DECLARAR FUNCIONES
def respuestas(argumento):
    switcher = {
    0: "inmaduro",
    1: "intermedio",
    2: "maduro",
    3: "no mango",
    4: "pasado"
    }
    return switcher.get(argumento, "inválido")
def predecir(file):
    x = load_img(file,target_size=(longitud,altura))
    x = img_to_array(x)
    x = np.expand_dims(x,axis=0)
    arreglo = cnn.predict(x) #retorna arreglo de 2 dimensiones
    resultado = arreglo[0] #dimensión 1 tiene la predicción
    respuesta = np.argmax(resultado)
    #print respuestas(respuesta)
    return respuestas(respuesta)
imagen1 = predecir("mango1.jpg")
print "imagen 1: ", imagen1
imagen2 = predecir("mango2.jpg")
print "imagen 2: ", imagen2
if(imagen1=="inmaduro" or imagen2=="inmaduro"): print "es inmaduro"
elif(imagen1=="pasado" or imagen2=="pasado"): print "es pasado"
else: print "es maduro"
#respuesta = predecir("mango2.jpg")
#print respuesta

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
    0: "inmaduro buena calidad",
    1: "inmaduro mala calidad",
    2: "intermedio buena calidad",
    3: "intermedio mala calidad",
    4: "maduro buena calidad",
    5: "maduro mala calidad",
    6: "no mango",
    7: "pasado"
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
imagen1 = predecir("fresa1.jpg")
#print "imagen 1: ", imagen1
imagen2 = predecir("fresa2.jpg")
#print "imagen 2: ", imagen2
if(imagen1=="inmaduro buena calidad" and imagen2=="inmaduro buena calidad"): print "inmaduro buena calidad"
elif(imagen1=="inmaduro mala calidad" and imagen2=="inmaduro mala calidad"): print "inmaduro mala calidad"
elif(imagen1=="inmaduro mala calidad" and imagen2=="inmaduro buena calidad"): print "inmaduro mala calidad"
elif(imagen1=="inmaduro buena calidad" and imagen2=="inmaduro mala calidad"): print "inmaduro mala calidad"
elif(imagen1=="pasado" and imagen2=="pasado"): print "pasado"
elif(imagen1=="maduro mala calidad" and imagen2=="maduro mala calidad"): print "maduro mala calidad"
elif(imagen1=="maduro buena calidad" and imagen2=="maduro buena calidad"): print "maduro buena calidad"
elif(imagen1=="maduro mala calidad" and imagen2=="maduro buena calidad"): print "maduro mala calidad"
elif(imagen1=="maduro buena calidad" and imagen2=="maduro mala calidad"): print "maduro mala calidad"
elif(imagen1=="maduro buena calidad" and imagen2=="pasado buena calidad"): print "pasado buena calidad"
elif(imagen2=="maduro buena calidad" and imagen1=="pasado buena calidad"): print "pasado buena calidad"
elif(imagen1=="maduro mala calidad" and imagen2=="pasado mala calidad"): print "pasado mala calidad"
elif(imagen1=="maduro mala calidad" and imagen2=="pasado buena calidad"): print "pasado mala calidad"
elif(imagen1=="maduro buena calidad" and imagen2=="pasado mala calidad"): print "pasado mala calidad"
elif(imagen2=="maduro mala calidad" and imagen1=="pasado mala calidad"): print "pasado"
elif(imagen1=="maduro buena calidad" and imagen2=="inmaduro buena calidad"): print "intermedio buena calidad"
elif(imagen1=="inmaduro buena calidad" and imagen2=="maduro buena calidad"): print "intermedio buena calidad"
elif(imagen1=="inmaduro mala calidad" and imagen2=="maduro mala calidad"): print "intermedio mala calidad"
elif(imagen1=="inmaduro buena calidad" and imagen2=="maduro mala calidad"): print "intermedio mala calidad"
elif(imagen1=="inmaduro mala calidad" and imagen2=="maduro buena calidad"): print "intermedio mala calidad"
elif(imagen2=="inmaduro mala calidad" and imagen1=="maduro buena calidad"): print "intermedio mala calidad"
elif(imagen1=="inmaduro mala calidad" and imagen2=="maduro mala calidad"): print "intermedio mala calidad"
elif(imagen2=="inmaduro mala calidad" and imagen1=="maduro mala calidad"): print "intermedio mala calidad"
elif(imagen1=="intermedio buena calidad" and imagen2=="maduro buena calidad"): print "maduro buena calidad"
elif(imagen2=="intermedio mala calidad" and imagen1=="maduro buena calidad"): print "maduro mala calidad"
elif(imagen1=="intermedio mala calidad" and imagen2=="intermedio buena calidad"): print "intermedio mala calidad"
elif(imagen2=="intermedio mala calidad" and imagen1=="intermedio buena calidad"): print "intermedio mala calidad"
elif(imagen1=="maduro buena calidad" and imagen2=="intermedio buena calidad"): print "maduro buena calidad"
elif(imagen2=="maduro mala calidad" and imagen1=="intermedio buena calidad"): print "maduro mala calidad"

elif(imagen1=="intermedio mala calidad" and imagen2=="maduro mala calidad"): print "maduro mala calidad"
elif(imagen1=="intermedio buena calidad" and imagen2=="maduro buena calidad"): print "maduro buena calidad"
elif(imagen2=="intermedio buena calidad" and imagen1=="maduro mala calidad"): print "maduro mala calidad"
elif(imagen2=="intermedio mala calidad" and imagen1=="maduro buena calidad"): print "maduro mala calidad"
elif(imagen1=="intermedio buena calidad" and imagen2=="maduro mala calidad"): print "maduro mala calidad"
elif(imagen1=="intermedio mala calidad" and imagen2=="maduro buena calidad"): print "maduro mala calidad"

elif(imagen1=="intermedio mala calidad" and imagen2=="inmaduro mala calidad"): print "inmaduro mala calidad"
elif(imagen1=="intermedio buena calidad" and imagen2=="inmaduro buena calidad"): print "inmaduro buena calidad"
elif(imagen2=="intermedio mala calidad" and imagen1=="inmaduro mala calidad"): print "inmaduro mala calidad"
elif(imagen2=="intermedio buena calidad" and imagen1=="inmaduro buena calidad"): print "inmaduro buena calidad"
elif(imagen1=="intermedio mala calidad" and imagen2=="inmaduro buena calidad"): print "inmaduro mala calidad"
elif(imagen1=="intermedio buena calidad" and imagen2=="inmaduro mala calidad"): print "inmaduro mala calidad"
elif(imagen1=="pasado" and imagen2=="inmaduro mala calidad"): print "pasado"
elif(imagen2=="pasado" and imagen1=="inmaduro buena calidad"): print "pasado"
elif(imagen1=="no mango" or imagen2=="no mango"): print "no mango"

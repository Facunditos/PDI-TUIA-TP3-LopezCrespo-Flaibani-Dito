import os
#from dados import *
from dados import imshow, imreconstruct, imfillhole, lectura, recortarxcontorno, detectar_contornos_cuadrados, es_cuadrado, procesar_color, redimensionar, contarDados, dado_quieto, programa_dados, grabar_video


# Obtener la lista de archivos de imágenes en el directorio actual
archivos_video = [f for f in os.listdir('./') if f.endswith(('.mp4'))]


for video in archivos_video:
  programa_dados(video)
  grabar_video(video)
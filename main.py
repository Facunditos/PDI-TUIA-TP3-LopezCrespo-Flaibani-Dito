import os
#from dados import *
from dados import imshow, imreconstruct, imfillhole, lectura, recortarxcontorno, detectar_contornos_cuadrados, es_cuadrado, procesar_color, redimensionar, contarDados, dado_quieto, programa_dados, grabar_video


# Obtener la lista de archivos de video (archivos .mp4) en el directorio actual; Busca en el directorio actual ('./') todos los archivos que terminen con la extensión .mp4 y los agrega a una lista llamada archivos_video.
""" Asegúrate de que todos los archivos .mp4 están en el directorio correcto:
El código utiliza os.listdir('./') para listar los archivos en el directorio donde se ejecuta el script. 
Si tus archivos de video .mp4 están en un subdirectorio, debes cambiar la ruta o asegurarte de estar ejecutando el script desde el directorio correcto."""
archivos_video = [f for f in os.listdir('./') if f.endswith(('.mp4'))] # [elemento for elemento in iterable if condición] lista por comprensión
                                            # f.endswith(('.mp4')): Para cada archivo f en la lista de archivos obtenida de os.listdir('./'), 
                                            # se verifica si nombre del archivo  termina con la extensión .mp4 usando el método endswith(). 
                                            # Este método devuelve True si la cadena f termina con .mp4 y False si no.

# Iterar sobre los videos y aplicar las funciones
for video in archivos_video:
  print(f"Procesando el video: {video}")
  programa_dados(video)
  grabar_video(video)
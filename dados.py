import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# os.makedirs("frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.
#-------------------
# Funciones
#-------------------
'''
Muestra imágenes por pantalla.
'''
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False)-> None:
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

'''
Reconstrucción Morgológica.
'''
def imreconstruct(marker: np.ndarray, mask: np.ndarray, kernel=None)-> np.ndarray:
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection

'''
Version 1
Utilizando reconstrucción morfológica
NO rellena los huecos que tocan los bordes
'''
def imfillhole(img: np.ndarray)-> np.ndarray:
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


def lectura(video):
    output_folder = "frames_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Abrir el video
    cap1 = cv2.VideoCapture('tirada_1.mp4') # Crea un objeto VideoCapture para abrir el archivo de video llamado 'tirada_1.mp4'.
    # Verificar si el video se abre correctamente
    if not cap1.isOpened(): # Verifica si el archivo de video se abrió correctamente. Si no se puede abrir el archivo, imprime un mensaje de error y termina la ejecución con exit().
        print("Error al abrir el video o la cámara") # Usar cap.isOpened() es una forma de asegurar que el programa puede continuar con la captura de video sin errores, asegurando que la fuente de video esté disponible.
        exit()
    frame_count = 0  # Contador de frames
    while True:
        # Leer un frame
        ret, frame = cap1.read()
        if not ret:
            break  # Si no hay más frames, salir del bucle
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1  # Aumentar el contador de frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap1.release()         # Liberar el objeto VideoCapture y
    cv2.destroyAllWindows() #  cerrar las ventanas
    print(f"Se han guardado {frame_count} frames en la carpeta '{output_folder}'.")
    return output_folder

def redimensionar(frame):
    height, width = frame.shape[:2] # frame.shape retorna una tupla que contiene (alto, ancho, canales), [:2] selecciona los dos primeros elementos de la tupla.
    frame = cv2.resize(frame, dsize=(int(width/2), int(height/2))) # El ancho se reduce a la mitad: int(width/2).
    return frame

def procesar_color(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Convierte la imagen de formato BGR
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255 
    h, s, v = cv2.split(img_hsv)      # División: Separa los canales de H, S, y V en matrices individuales usando cv2.split.
    ix_h1 = np.logical_and(h > 179 * .9, h < 179 * 0.1) # Condiciones para H (Tono): ix_h1: Detecta tonos de rojo cercanos al valor superior del espectro (entre 90% y 100% de 180).
    ix_h2 = h < 180 * 0.04                        # ix_h2: Detecta tonos de rojo cercanos al valor inferior del espectro (0 a 4% de 180).
    ix_s = np.logical_and(s > 255 * 0.3, s < 255)   # Condiciones para S (Saturación): ix_s: Detecta colores con saturación entre el 30% y el 100% de 255.
    ix = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s) # Máscara combinada (ix): Unión lógica: Combina las máscaras de tonos de rojo (ix_h1 o ix_h2) con la de saturación (ix_s)

    # Eliminar colores no deseados:
    r, g, b = cv2.split(img) # Se separan los canales RGB
    r[ix != True] = 0 # Los píxeles que no cumplen con la máscara (ix) se eliminan (se ponen a 0).
    g[ix != True] = 0
    b[ix != True] = 0
    # Reconstruir la imagen roja
    rojo_img = cv2.merge((r, g, b)) # e combinan los canales para obtener la imagen con el color rojo resaltado.
    img_gris =  cv2.cvtColor(rojo_img, cv2.COLOR_RGB2GRAY) # a imagen procesada se convierte a escala de grises, útil para análisis o detección adicional.
    return img_gris,rojo_img # rojo_img: Imagen RGB con solo los píxeles rojos. 
                             # img_gris: Imagen en escala de grises con el color rojo resaltado.

def es_cuadrado(contorno):
    épsilon = 0.05* cv2.arcLength (contorno, True )
    perimetro = cv2.arcLength(contorno, True)  # Calcula el perímetro del contorno (True indica que el contorno es cerrado,que forma un polígono cerrado,  lo que significa que se conecta en su punto final con su punto inicial.)
    approx = cv2.approxPolyDP(contorno, épsilon, True) # Aproxima el contorno a un polígono, Si el contorno es cuadrado, la aproximación tendrá 4 vértices.
    if len(approx) == 4:  # Verifica si el contorno tiene 4 vértices, lo que indica que es un cuadrado o podria ser un rectangulo
        # Verificar si los ángulos son aproximadamente 90 grados
        for i in range(4):
           for i in range(4): # Este ciclo recorre los 4 vértices de la aproximación (almacenados en approx).
            p1 = approx[i][0]  # Primer vértice
            p2 = approx[(i + 1) % 4][0]  # Segundo vértice (el siguiente, con índice cíclico)
            p3 = approx[(i + 2) % 4][0]  # Tercer vértice (el siguiente después de p2)
            # Calcular el vector del primer lado (de p1 a p2)
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Calcular el vector del segundo lado (de p2 a p3)
            vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            # Calcular el producto punto entre los dos vectores
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            # Calcular la magnitud de los vectores (longitudes)
            magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2) # Magnitud del primer vector
            magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)  # Magnitud del segundo vector
            # Calcular el coseno del ángulo entre los dos vectores
            cos_theta = dot_product / (magnitude1 * magnitude2) # Coseno del ángulo entre los vectores
            # Si el coseno es cercano a 0, significa que los vectores son perpendiculares, Si el coseno del ángulo no es cercano a cero, 
            # significa que los vectores no son perpendiculares (es decir, el ángulo no es cercano a 90 grados).
            if abs(cos_theta) > 0.1:
                return False  # Si los ángulos no son cercanos a 90 grados, no es un cuadrado
        return True  # Si todos los ángulos son aproximadamente 90 grados, es un cuadrado
    return False  # Si no tiene 4 vértices, no es cuadrado


# Función para detectar contornos cuadrados 
def detectar_contornos_cuadrados(frame): # tiene como objetivo identificar y devolver una lista de contornos que representan formas cuadradas en una imagen.
    frame_suavizado = cv2.GaussianBlur(frame, (5, 5), 0)  # Esto aplica un filtro gaussiano con un kernel de 5×5 para reducir el ruido sin perder demasiados detalles.
    # Umbralización Adaptativa: Aplicar umbral adaptativo para resaltar contornos 
    _, umbral = cv2.threshold(frame_suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Esto convierte la imagen en binaria usando el método de Otsu, separando claramente los objetos del fondo.
    # Detección de Contornos:: Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra los contornos externos de los objetos en la imagen binaria.
    # Filtrar contornos cuadrados aproximados
    contornos_cuadrados = [
        cnt for cnt in contornos if es_cuadrado(cnt) and cv2.contourArea(cnt) > 50*50
    ] # Área Mínima: Excluye contornos con áreas menores a 50×50 píxeles para evitar detectar ruido.
    return contornos_cuadrados

# FUNCION RECORTAR CONTORNOS 
contornos = detectar_contornos_cuadrados(frame)
def recortarxcontorno(frame, contornos):
    mask = np.zeros_like(frame)
    recortes = []
    for contorno in contornos:
        mask = np.zeros_like(frame)
        # Dibuja el contorno en la máscara
        cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
        # Aplica la máscara a la imagen original
        dado_recortado = cv2.bitwise_and(frame,frame, mask=mask)
        #cv2.imshow('Solo contorno',redimensionar(dado_recortado))
        recortes.append(dado_recortado)
    return recortes

mask = np.zeros_like(frame)
recortes = []
for contorno in contornos:
    mask = np.zeros_like(frame)
    # Dibuja el contorno en la máscara
    cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
    # Aplica la máscara a la imagen original
    dado_recortado = cv2.bitwise_and(frame,frame, mask=mask)
    #cv2.imshow('Solo contorno',redimensionar(dado_recortado))
    recortes.append(dado_recortado)


####################################################################################################################################################
                                                                    ## PRUEBAS - DESGLOSANDO CÓDIGO ##
####################################################################################################################################################

video = "tirada_1.mp4"
cap = cv2.VideoCapture(video)  # Capturar video desde un archivo:
ret, frame = cap.read()
ret   #  Booleano que indica si la operación de lectura fue exitosa.
frame  # El frame leído del video, imagen en forma de arreglo NumPy que contiene los valores de color de cada píxel, representa un solo fotograma (o imagen) del video.
""" frame: Es una matriz de alto x ancho x 3 (en caso de imágenes en color) que representa los píxeles del fotograma. 
Cada uno de los tres canales de color (Rojo, Verde, Azul, en formato RGB o BGR dependiendo de la configuración de OpenCV) 
tiene valores de intensidad entre 0 y 255.
Por ejemplo, si tienes una imagen de tamaño 640x480, frame será una matriz de 
(480, 640, 3) (si es RGB o BGR), donde 480 es la altura de la imagen, 640 es el ancho y 3 corresponde a los tres canales de color."""

# info del video
fps = cap.get(cv2.CAP_PROP_FPS)  # las frames por segundo. # 30
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # el número total de frames en el video. 146
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 1080
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 2224
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# Liberar el objeto cap cuando hayas terminado: Después de terminar con la captura de video, es importante liberar el objeto para liberar los recursos.
cap.release()

# PRUEBA 1:
"""
Resumen de los pasos:
    1.Crear una carpeta de salida: Se verifica si existe una carpeta llamada frames_output donde se guardarán los frames extraídos. Si no existe, se crea automáticamente.
    2.Abrir el video: Se utiliza cv2.VideoCapture() para abrir el archivo de video especificado, en este caso 'tirada_1.mp4'.
    3.Verificación de apertura exitosa: Se comprueba si el video se ha abierto correctamente usando cap1.isOpened(). Si no se pudo abrir el video, el programa se detiene.
    4.Leer y procesar cada frame: Se entra en un bucle infinito donde se leen los frames del video uno a uno usando cap1.read().
    5.Si no se puede leer un frame, el bucle se interrumpe.
    6.Guardar los frames en una carpeta: Para cada frame leído, se guarda la imagen en la carpeta frames_output utilizando cv2.imwrite().
    7.Los frames se guardan con nombres consecutivos (por ejemplo, frame_0.jpg, frame_1.jpg, etc.).
    8.Control de cuántos frames se procesan: Solo se guardan los frames según un intervalo determinado por la variable frame_skip. En el ejemplo, se guarda 1 frame cada 5 frames leídos (es decir, frame_skip = 5).
    9.Control de salida: El código permite salir del bucle si el usuario presiona la tecla 'q'. Esta parte es opcional y sirve para salir del procesamiento si se desea antes de que termine el video.
    10.Liberar recursos: Después de procesar los frames, se libera el objeto cap1 y se cierran todas las ventanas de OpenCV con cv2.destroyAllWindows()."""

# Crear una carpeta donde se guardarán los frames si no existe
output_folder = "frames_output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Abrir el video
cap1 = cv2.VideoCapture('tirada_1.mp4') # Crea un objeto VideoCapture para abrir el archivo de video llamado 'tirada_1.mp4'.

# Verificar si el video se abre correctamente
if not cap1.isOpened(): # Verifica si el archivo de video se abrió correctamente. Si no se puede abrir el archivo, imprime un mensaje de error y termina la ejecución con exit().
    print("Error al abrir el video o la cámara") # Usar cap.isOpened() es una forma de asegurar que el programa puede continuar con la captura de video sin errores, asegurando que la fuente de video esté disponible.
    exit()

frame_count = 0  # Contador de frames

while True:
    # Leer un frame
    ret, frame = cap1.read()
    
    if not ret:
        break  # Si no hay más frames, salir del bucle

    # Guardar el frame en la carpeta como una imagen .jpg
    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1  # Aumentar el contador de frames

    # Opcional: Si deseas mostrar el frame, puedes hacerlo, pero puede ser lento
    # cv2.imshow('Frame', frame)

    # Para controlar la velocidad de lectura de los frames (puedes ajustar el retardo)
    # Si deseas salir presionando la tecla 'q':
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """ Controlar la tasa de frames: Si el video es muy rápido o grande, se puede ralentizar un poco el procesamiento de los frames 
        utilizando un valor mayor en cv2.waitKey(), por ejemplo, 30 milisegundos:"""
    #if cv2.waitKey(30) & 0xFF == ord('q'):
       # break


# Estas líneas solo se deben llamar después de que hayas terminado con la captura de video y el proceso de visualización.
cap1.release()         # Liberar el objeto VideoCapture y
cv2.destroyAllWindows() #  cerrar las ventanas

print(f"Se han guardado {frame_count} frames en la carpeta '{output_folder}'.")

# convierto a FUNCION Lectura ##########################################################################################
def lectura(video):
    output_folder = "frames_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Abrir el video
    cap1 = cv2.VideoCapture('tirada_1.mp4') # Crea un objeto VideoCapture para abrir el archivo de video llamado 'tirada_1.mp4'.

    # Verificar si el video se abre correctamente
    if not cap1.isOpened(): # Verifica si el archivo de video se abrió correctamente. Si no se puede abrir el archivo, imprime un mensaje de error y termina la ejecución con exit().
        print("Error al abrir el video o la cámara") # Usar cap.isOpened() es una forma de asegurar que el programa puede continuar con la captura de video sin errores, asegurando que la fuente de video esté disponible.
        exit()

    frame_count = 0  # Contador de frames

    while True:
        # Leer un frame
        ret, frame = cap1.read()
        
        if not ret:
            break  # Si no hay más frames, salir del bucle

        # Guardar el frame en la carpeta como una imagen .jpg
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1  # Aumentar el contador de frames

        # Opcional: Si deseas mostrar el frame, puedes hacerlo, pero puede ser lento
        # cv2.imshow('Frame', frame)

        # Para controlar la velocidad de lectura de los frames (puedes ajustar el retardo)
        # Si deseas salir presionando la tecla 'q':
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """ Controlar la tasa de frames: Si el video es muy rápido o grande, se puede ralentizar un poco el procesamiento de los frames 
            utilizando un valor mayor en cv2.waitKey(), por ejemplo, 30 milisegundos:"""
        #if cv2.waitKey(30) & 0xFF == ord('q'):
        # break


    # Estas líneas solo se deben llamar después de que hayas terminado con la captura de video y el proceso de visualización.
    cap1.release()         # Liberar el objeto VideoCapture y
    cv2.destroyAllWindows() #  cerrar las ventanas

    print(f"Se han guardado {frame_count} frames en la carpeta '{output_folder}'.")
    return output_folder

output_folder = lectura('tirada_1.mp4')

# funcion redimensionar tamaño para ahorra costo computacional ##################################################################################3
"""
# Funcion para REDIMENSIONAR tamaños :  útil para 1-optimización de recursos: Reducir el tamaño de una imagen o video para ahorrar memoria y acelerar cálculos.
                                                # 2.Preprocesamiento: Preparar imágenes para modelos que requieren entradas de tamaño fijo.
# La imagen es la misma pero más pequeña: Todo el contenido se reduce en tamaño. 
# Mantiene todo el contenido de la imagen:
La proporción y los objetos en la imagen no cambian; simplemente se ajusta la cantidad de píxeles que la representan.
No selecciona ni recorta ninguna parte específica de la imagen.   
Al reducir la imagen ( a la mitad), OpenCV utiliza métodos de interpolación para eliminar píxeles y comprimir la imagen.
NOTA: Aunque se reduce el tamaño, cuando la visualizas, el visor o librería (como matplotlib) puede escalar automáticamente la imagen 
    redimensionada para ajustarla al tamaño de la ventana. Por eso parece que ves la misma imagen.
"""                                       
def redimensionar(frame):
    height, width = frame.shape[:2] # frame.shape retorna una tupla que contiene (alto, ancho, canales), [:2] selecciona los dos primeros elementos de la tupla.
    frame = cv2.resize(frame, dsize=(int(width/2), int(height/2))) # El ancho se reduce a la mitad: int(width/2).
    return frame

# Redimensionar usando la función
frame= cv2.imread('frames_output/frame_80.jpg') # OpenCV carga imágenes en el espacio de color BGR, pero matplotlib espera imágenes en el formato RGB. Por eso, usamos:
frame_reducido = redimensionar(frame)

# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convertir de BGR a RGB
plt.title("Imagen Original")
#plt.axis('off')  # Ocultar ejes
plt.axis()

# Imagen redimensionada
plt.subplot(1, 2, 2)                      # Convertir de BGR a RGB para matplotlib. 
plt.imshow(cv2.cvtColor(frame_reducido, cv2.COLOR_BGR2RGB)) #  plt.imshow() renderiza la imagen: Esto es necesario antes de llamar a plt.show().
plt.axis()  # Ocultar ejes
plt.title("Imagen Redimensionada")

# Mostrar las imágenes
plt.tight_layout()  # Ajustar el diseño para evitar solapamiento
plt.show()

# Mostrar los tamaños
print("Tamaño original:", frame.shape[:2]) # (2224, 1080)
print("Tamaño redimensionado:", frame_reducido.shape[:2]) # (1112, 540)

# funcion PROCESAR COLOR ###################################################################################################################
# Seguimiento de objetos rojos en videos.
# Segmentación de colores específicos para análisis
# Opcion 1
def procesar_color(frame):
    """ La función procesar_color realiza operaciones de segmentación y transformación de una imagen para detectar y 
        resaltar el color rojo en un frame (imagen), y devuelve:
            La imagen segmentada en escala de grises.
            La imagen con el color rojo resaltado en formato RGB."""

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Convierte la imagen de formato BGR
    """ Convierte la imagen a espacio de color HSV:
                    H (Tono): Describe el color (valores entre 0 y 179).
                    S (Saturación): Intensidad del color (0-255).
                    V (Valor): Brillo (0-255). """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255 
    h, s, v = cv2.split(img_hsv)      # División: Separa los canales de H, S, y V en matrices individuales usando cv2.split.
 

    # Segmentacion en color - Detectar solo el rojo 
    """El rojo en HSV generalmente tiene dos rangos:
        Rango bajo: H de 0 a 10.
        Rango alto: H de 160 a 179.
    """
    ix_h1 = np.logical_and(h > 179 * .9, h < 179 * 0.1) # Condiciones para H (Tono): ix_h1: Detecta tonos de rojo cercanos al valor superior del espectro (entre 90% y 100% de 180).
    ix_h2 = h < 180 * 0.04                        # ix_h2: Detecta tonos de rojo cercanos al valor inferior del espectro (0 a 4% de 180).
    ix_s = np.logical_and(s > 255 * 0.3, s < 255)   # Condiciones para S (Saturación): ix_s: Detecta colores con saturación entre el 30% y el 100% de 255.
    ix = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s) # Máscara combinada (ix): Unión lógica: Combina las máscaras de tonos de rojo (ix_h1 o ix_h2) con la de saturación (ix_s)
    # ix2 = (ix_h1 | ix_h2) & ix_s   # Otra opcion que da igual...
    
    # Eliminar colores no deseados:
    r, g, b = cv2.split(img) # Se separan los canales RGB
    r[ix != True] = 0 # Los píxeles que no cumplen con la máscara (ix) se eliminan (se ponen a 0).
    g[ix != True] = 0
    b[ix != True] = 0
    # Reconstruir la imagen roja
    rojo_img = cv2.merge((r, g, b)) # e combinan los canales para obtener la imagen con el color rojo resaltado.

    img_gris =  cv2.cvtColor(rojo_img, cv2.COLOR_RGB2GRAY) # a imagen procesada se convierte a escala de grises, útil para análisis o detección adicional.

    return img_gris,rojo_img # rojo_img: Imagen RGB con solo los píxeles rojos. 
                             # img_gris: Imagen en escala de grises con el color rojo resaltado.


# Procesar la imagen
img_gris, rojo_img = procesar_color(frame_reducido) # Ojo agarrar un frame que tenga dados

# Mostrar resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Escala de Grises")
plt.imshow(img_gris, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Rojo Resaltado")
plt.imshow(cv2.cvtColor(rojo_img, cv2.COLOR_RGB2BGR))  # Convertir a BGR para visualizar correctamente
plt.show()

## Opcion 2
def procesar_color1(frame):
    """Segmenta y resalta el color rojo en una imagen, devolviendo una versión en escala de grises y otra RGB."""
    
    # Convertir a RGB y luego a HSV
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Definir rangos de rojo en HSV
    lower_red_1 = np.array([0, 50, 50])    # Rango bajo del rojo De H = 0 a 10 (rojo "bajo").
    upper_red_1 = np.array([10, 255, 255])  
    lower_red_2 = np.array([160, 50, 50])  # Rango alto del rojo De H = 160 a 179 (rojo "alto").
    upper_red_2 = np.array([179, 255, 255])

    # Crear máscaras para cada rango
    mask1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1) # Esto crea una máscara binaria que marca los píxeles dentro del rango especificado.
    mask2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)

    # Combinar máscaras
    mask = cv2.bitwise_or(mask1, mask2)  # cv2.bitwise_or combina las máscaras para incluir ambos rangos de rojo.

    # Aplicar la máscara a la imagen original
    rojo_img = cv2.bitwise_and(img, img, mask=mask) # cv2.bitwise_and conserva solo los píxeles marcados en la máscara y elimina el resto.

    # Convertir a escala de grises
    img_gris = cv2.cvtColor(rojo_img, cv2.COLOR_RGB2GRAY) # Se genera una imagen en escala de grises con solo los píxeles rojos resaltados.

    return img_gris, rojo_img

# Procesar la imagen
img_gris, rojo_img = procesar_color1(frame_reducido) # Ojo agarrar un frame que tenga dados

# Mostrar resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Escala de Grises")
plt.imshow(img_gris, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Rojo Resaltado")
plt.imshow(cv2.cvtColor(rojo_img, cv2.COLOR_RGB2BGR))  # Convertir a BGR para visualizar correctamente
plt.show()

#----------------------------------------------------------------
"""¿Cómo encontrar valores HSV para realizar seguimiento?
Esta es una pregunta común que se encuentra en stackoverflow.com . Es muy simple y puedes usar la misma función, cv.cvtColor() . En lugar de pasar una imagen, 
simplemente pasa los valores BGR que deseas. Por ejemplo, para encontrar el valor HSV de red"""
video = "tirada_1.mp4"
cap = cv2.VideoCapture(video)  # Capturar video desde un archivo:
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
verde = np.uint8([[[0,255,0]]]) # espacio de color BGR (utilizado por OpenCV): 0: Intensidad del azul.
                                                                             # 255: Máxima intensidad del verde.
                                                                             # 0: Intensidad del rojo.
# OpenCV utiliza el formato BGR por defecto, no RGB. Así que este arreglo corresponde a una imagen de 1 píxel donde solo el componente verde está activo
#rojo = np.uint8([[[0, 0, 255]]]) # rojo

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_verde = cv2.cvtColor(verde,cv2.COLOR_BGR2HSV) # Blue, green, Red = BGR
print(hsv_verde) # [[[60 255 255]]]
#Ahora toma [H-10, 100,100] y [H+10, 255, 255] como límite inferior y límite superior respectivament


# Crear un píxel verde y rojo en formato BGR
azul = np.uint8([[[255, 0, 0]]])   # azul en BGR
verde = np.uint8([[[0, 255, 0]]])  # Verde en BGR
rojo = np.uint8([[[0, 0, 255]]])   # Rojo en BGR

# Convertir ambos al espacio HSV
azul_hsv = cv2.cvtColor(azul, cv2.COLOR_BGR2HSV)
verde_hsv = cv2.cvtColor(verde, cv2.COLOR_BGR2HSV)
rojo_hsv = cv2.cvtColor(rojo, cv2.COLOR_BGR2HSV)

print(f"Azul en HSV: {azul_hsv[0][0]}") # [120 255 255]   120 en OpenCV ≈ 240° (Azul).
print(f"Verde en HSV: {verde_hsv[0][0]}") # [ 60 255 255]  60 en OpenCV ≈ 120° (Verde).
print(f"Rojo en HSV: {rojo_hsv[0][0]}") # [  0 255 255]    0 en OpenCV = 0° (Rojo).

"""
El círculo cromático es una representación visual de los colores dispuestos en un círculo, que muestra las relaciones entre ellos. 
En el espacio de color HSV utilizado por OpenCV, el H (Hue o matiz) determina la posición del color en este círculo cromático.

Concepto Básico del Círculo Cromático
    Hue (Matiz): Representa el color puro y se mide en grados de un círculo cromático (0° a 360°).
    0°:  Rojo.
    120°: Verde.
    240°: Azul.

En OpenCV, este rango está escalado de 0 a 179 para optimización:
    0 en OpenCV = 0° (Rojo).
    60 en OpenCV ≈ 120° (Verde).
    120 en OpenCV ≈ 240° (Azul).

La representación completa del círculo cromático en OpenCV queda como:
    0-29: Rojo.
    30-89: Amarillo a verde.
    90-149: Verde a cian.
    150-179: Azul a magenta."""

# Crear un círculo cromático --------------------------------------------------------------------------------
hue = np.linspace(0, 179, 180, dtype=np.uint8)  # Valores de H
hue = np.tile(hue, (180, 1))  # Expandir en 2D
sat = np.full_like(hue, 255)  # Saturación máxima
val = np.full_like(hue, 255)  # Brillo máximo

hsv_circle = cv2.merge([hue, sat, val])  # Combinar en espacio HSV
rgb_circle = cv2.cvtColor(hsv_circle, cv2.COLOR_HSV2RGB)  # Convertir a RGB

plt.imshow(rgb_circle)
plt.title("Círculo Cromático HSV")
plt.axis('off')
plt.show()



## FUNCION CUADRADOS ##################################################################################################################
# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
"""TEORIA OpencV
img = cv.imread('star.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
 
cnt = contours[0]
M = cv2.moments(cnt)
print( M )

A partir de estos momentos, puedes extraer datos útiles como área, centroide, etc:
    *El centroide se proporciona mediante las relaciones:
cx = int(M[ 'm10' ]/M[ 'm00' ])
cy = int(M[ 'm01' ]/M[ 'm00' ])

El área del contorno se proporciona mediante la función cv.contourArea() o a partir de los momentos, M['m00'] :
área = cv2.áreaContorno (cnt)

   *Perímetro del contorno
También se denomina longitud de arco. Se puede averiguar mediante la función cv.arcLength() . 
El segundo argumento especifica si la forma es un contorno cerrado (si se pasa como True) o simplemente una curva.
perímetro = cv2.arcLength (cnt, True )

    * Aproximación de contornos
Aproxima una forma de contorno a otra forma con un número menor de vértices según la precisión que especifiquemos. 
Es una implementación del algoritmo de Douglas-Peucker . Consulte la página de Wikipedia para ver el algoritmo y la demostración.

Para entender esto, supongamos que estás intentando encontrar un cuadrado en una imagen, pero debido a algunos problemas en la imagen,
 no obtuviste un cuadrado perfecto, sino una "mala forma" (como se muestra en la primera imagen a continuación). Ahora puedes usar esta función para aproximar la forma. En este caso, el segundo argumento se llama épsilon, que es la distancia máxima desde el contorno hasta el contorno aproximado. 
Es un parámetro de precisión. Se necesita una selección inteligente de épsilon para obtener el resultado correcto.
épsilon = 0,1* cv2.arcLength (cnt, True )
aprox = cv2.approxPolyDP (cnt,épsilon, Verdadero )
"""

def es_cuadrado(contorno):
    """ La función es_cuadrado(contorno) detecta si un contorno tiene una forma aproximada de cuadrado basándose en el número de vértices del contorno. 
    Si el contorno tiene 4 vértices, devuelve True, sugiriendo que el contorno es cuadrado o rectangular. Si el contorno tiene más o menos de 4 vértices, devuelve False.
    épsilon = 0,1* cv.arcLength (cnt, True )
    aprox = cv.approxPolyDP (cnt,épsilon, Verdadero )"""
    épsilon = 0.05* cv2.arcLength (contorno, True )
    perimetro = cv2.arcLength(contorno, True)  # Calcula el perímetro del contorno (True indica que el contorno es cerrado,que forma un polígono cerrado,  lo que significa que se conecta en su punto final con su punto inicial.)
    approx = cv2.approxPolyDP(contorno, épsilon, True) # Aproxima el contorno a un polígono, Si el contorno es cuadrado, la aproximación tendrá 4 vértices.
    """  es la precisión del algoritmo. El valor 0.05 * perimetro indica cuánto se puede reducir el número de vértices del contorno sin que se pierda mucha información. 
    Un valor de 0.05 es un valor comúnmente usado para preservar la forma general sin perder demasiados detalles, cuanto menor sea este valor, más puntos mantendrá la aproximación."""
    if len(approx) == 4:  # Verifica si el contorno tiene 4 vértices, lo que indica que es un cuadrado o podria ser un rectangulo
        """Si deseas una verificación más precisa para asegurar que los contornos sean cuadrados, podrías agregar una comprobación adicional 
        para evaluar si los ángulos entre los lados del polígono aproximado son cercanos a 90 grados."""
        # Verificar si los ángulos son aproximadamente 90 grados
        for i in range(4):
           for i in range(4): # Este ciclo recorre los 4 vértices de la aproximación (almacenados en approx).
            """ p1, p2, y p3 son tres vértices consecutivos. Notarás que usamos (i + 1) % 4 y (i + 2) % 4 para hacer el acceso cíclico a los índices, es decir, al final de la lista de 
            vértices volvemos al principio. Esto es importante para comparar los lados del contorno de forma continua"""
            p1 = approx[i][0]  # Primer vértice
            p2 = approx[(i + 1) % 4][0]  # Segundo vértice (el siguiente, con índice cíclico)
            p3 = approx[(i + 2) % 4][0]  # Tercer vértice (el siguiente después de p2)

            # Calcular el vector del primer lado (de p1 a p2)
            vec1 = (p2[0] - p1[0], p2[1] - p1[1])
            # Calcular el vector del segundo lado (de p2 a p3)
            vec2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calcular el producto punto entre los dos vectores
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            # Calcular la magnitud de los vectores (longitudes)
            magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2) # Magnitud del primer vector
            magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)  # Magnitud del segundo vector
            
            # Calcular el coseno del ángulo entre los dos vectores
            cos_theta = dot_product / (magnitude1 * magnitude2) # Coseno del ángulo entre los vectores
            
            # Si el coseno es cercano a 0, significa que los vectores son perpendiculares, Si el coseno del ángulo no es cercano a cero, 
            # significa que los vectores no son perpendiculares (es decir, el ángulo no es cercano a 90 grados).
            if abs(cos_theta) > 0.1:
                return False  # Si los ángulos no son cercanos a 90 grados, no es un cuadrado

        return True  # Si todos los ángulos son aproximadamente 90 grados, es un cuadrado
    return False  # Si no tiene 4 vértices, no es cuadrado


# Función para detectar contornos cuadrados ------------------------------------------------------------------------------------
def detectar_contornos_cuadrados(frame): # tiene como objetivo identificar y devolver una lista de contornos que representan formas cuadradas en una imagen.
    """ Imagen suavizada pasa al umbral adaptativo.
        Contornos detectados en la imagen binaria.
        Filtrado para seleccionar solo contornos cuadrados.
    """
    # Suavizar la imagen para reducir el ruido
    frame_suavizado = cv2.GaussianBlur(frame, (5, 5), 0)  # Esto aplica un filtro gaussiano con un kernel de 5×5 para reducir el ruido sin perder demasiados detalles.
    # Umbralización Adaptativa: Aplicar umbral adaptativo para resaltar contornos 
    _, umbral = cv2.threshold(frame_suavizado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Esto convierte la imagen en binaria usando el método de Otsu, separando claramente los objetos del fondo.

    # Detección de Contornos:: Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra los contornos externos de los objetos en la imagen binaria.

    # Filtrar contornos cuadrados aproximados
    contornos_cuadrados = [
        cnt for cnt in contornos if es_cuadrado(cnt) and cv2.contourArea(cnt) > 50*50
    ] # Área Mínima: Excluye contornos con áreas menores a 50×50 píxeles para evitar detectar ruido.
    return contornos_cuadrados

# FUNCION RECORTAR CONTORNOS --------------------------------------------------------------------------------------
contornos = detectar_contornos_cuadrados(frame)
def recortarxcontorno(frame, contornos):
    mask = np.zeros_like(frame)
    recortes = []
    for contorno in contornos:
        mask = np.zeros_like(frame)
        # Dibuja el contorno en la máscara
        cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
        # Aplica la máscara a la imagen original
        dado_recortado = cv2.bitwise_and(frame,frame, mask=mask)
        #cv2.imshow('Solo contorno',redimensionar(dado_recortado))
        recortes.append(dado_recortado)
    return recortes

mask = np.zeros_like(frame)
recortes = []
for contorno in contornos:
    mask = np.zeros_like(frame)
    # Dibuja el contorno en la máscara
    cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
    # Aplica la máscara a la imagen original
    dado_recortado = cv2.bitwise_and(frame,frame, mask=mask)
    #cv2.imshow('Solo contorno',redimensionar(dado_recortado))
    recortes.append(dado_recortado)




#FUNCION CONTAR DADOS ----------------------------------------------------------------------------------------------------
def contarDados(recorte):
    _, umbral = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    # Aplicar erosión
    imagen_erosionada = cv2.erode(umbral, kernel, iterations=1)
    #cv2.imshow('Frame erode', redimensionar(imagen_erosionada))
    # Encuentra componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_erosionada, 4)

    # Especifica el umbral de área
    area_threshold = (20, 150)  # UMBRAL DE AREA

    # Filtra las componentes conectadas basadas en el umbral de área
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []

    # Mostrar la imagen con los puntos y bounding boxes
    #cv2.imshow('Puntos y Bounding Boxes', redimensionar(recorte))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #plt.imshow(recorte)
    for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
        area = stats[label, cv2.CC_STAT_AREA]

        if area > area_threshold[0] and area < area_threshold[1]:
            x, y, w, h, _ = stats[label]
            relacion_aspecto = float(w) / h
            if relacion_aspecto >=  0.7 and relacion_aspecto <= 1.3:
                filtered_labels.append(label)
                filtered_stats.append(stats[label])
                filtered_centroids.append(centroids[label])
            # Dibujar el bounding box
                #cv2.rectangle(recorte, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #cv2.rectangle(umbral, (x, y), (x + w, y + h), (255, 0, 0), 2)
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none'))
                #plt.imshow(recorte)
    #plt.show()

    return len(filtered_centroids)





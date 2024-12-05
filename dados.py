import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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


def create_green_mask(frame):
    """
    Detecta la zona del paño verde y genera una máscara binaria.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convierte el frame a espacio de color HSV
    # Definir límites para el color verde en HSV (ajusta según el video), define un rango (lower_green y upper_green).
    lower_green = np.array([35, 50, 50])  # H, S, V mínimos
    upper_green = np.array([85, 255, 255])  # H, S, V máximos
    # Crear máscara binaria del paño verde
    mask = cv2.inRange(hsv, lower_green, upper_green) # Usa cv2.inRange() para generar una máscara donde los píxeles en el rango definido se marcan como blancos.
    mask_fill = imfillhole(mask) # Llena agujeros en la máscara con una función adicional (imfillhole).
    return mask_fill


def create_red_mask(frame):
    """
    Detecta los dados rojos, los píxeles de color rojo en el frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convierte el frame a HSV.
    # Definir límites para el color rojo en HSV
    """ Define dos rangos de color rojo:
Rango 1 para tonos de rojo bajos (lower_red1, upper_red1).
Rango 2 para tonos de rojo altos (lower_red2, upper_red2).
Combina las máscaras de los dos rangos usando cv2.bitwise_or()."""
    lower_red1 = np.array([0, 50, 50])   # Rojo rango 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Rojo rango 2
    upper_red2 = np.array([180, 255, 255])
    # Máscaras para los dos rangos de rojo
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    return mask_red


def detect_red_dados(frame, mask_green):
    """
    Detecta los dados rojos dentro de la región delimitada por el paño verde.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Definir límites para el color rojo en HSV
    lower_red1 = np.array([0, 50, 50])   # Rojo rango 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # Rojo rango 2
    upper_red2 = np.array([180, 255, 255])
    # Máscaras para los dos rangos de rojo
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2) # Combina la máscara roja con la máscara verde usando cv2.bitwise_and()
    # Limitar la detección de dados al área del paño verde
    mask_dados = cv2.bitwise_and(mask_red, mask_green)
    # Buscar contornos en la máscara resultante
    contours, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Encuentra los contornos en la máscara combinada y calcula sus centroides.
    dados_coords = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filtrar ruido (ajusta el área mínima), Filtra contornos pequeños para evitar el ruido.
            # Obtener el centroide del contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dados_coords.append((cx, cy))
    return mask_dados, dados_coords # Devuelve: La máscara de los dados detectados, Una lista de coordenadas de los centroides de los dados.


def stopped(queue_coords, umbral=2):
    """
    Verifica si las coordenadas de los dados están quietas en la cola.
    Propósito: Verificar si los dados están quietos basándose en una cola de coordenadas recientes.
Lógica: Recorre la cola de coordenadas.
Compara las posiciones actuales y previas para verificar si el desplazamiento es menor que un umbral (2 píxeles por defecto).
Si algún dado se mueve más allá del umbral, devuelve False.
    """
    for i in range(1, len(queue_coords)):
        for (x1, y1), (x2, y2) in zip(queue_coords[i - 1], queue_coords[i]):
            if abs(x1 - x2) > umbral or abs(y1 - y2) > umbral:
                return False
    return True                 # Devuelve True si los dados están quietos; de lo contrario, False.


def get_dados_info(mask_dados, dados_coords, roi_size=100):
    """
    Genera un diccionario con la información de cada dado detectado, como su posición y número de pips.
    Args:
        mask_dados (np.ndarray): Máscara binaria de los dados.
        dados_coords (list): Coordenadas de los centroides de los dados.
        roi_size (int): Tamaño de la ROI alrededor del dado.
    Returns:
        list[dict]: Lista de diccionarios con información de cada dado.

    Lógica:
    Por cada dado:
        Define una Región de Interés (ROI) alrededor de su centroide.
        Invierte la máscara para resaltar los pips como blancos.
        Aplica operaciones morfológicas (cv2.morphologyEx) para limpiar ruido.
        Usa cv2.connectedComponentsWithStats() para contar los pips basándose en áreas.
        Filtra componentes pequeños para evitar detectar ruido.
    Resultado: Devuelve una lista de diccionarios con información de cada dado:
        Coordenadas del dado.
        ROI procesado.
        Número de pips.
    """
    dados_info = []
    for cx, cy in dados_coords:
        # Definir la ROI alrededor del centroide del dado
        x_start = max(cx - roi_size // 2, 0)
        x_end = min(cx + roi_size // 2, mask_dados.shape[1])
        y_start = max(cy - roi_size // 2, 0)
        y_end = min(cy + roi_size // 2, mask_dados.shape[0])
        roi = mask_dados[y_start:y_end, x_start:x_end]
        # Invertir la imagen para que los pips sean blancos
        roi_inverted = cv2.bitwise_not(roi)
        # Aplicar morfología para limpiar ruido
        kernel = np.ones((3, 3), np.uint8)
        roi_cleaned = cv2.morphologyEx(roi_inverted, cv2.MORPH_OPEN, kernel)
        # Detectar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_cleaned, connectivity=8)
        # Filtrar componentes por área
        pip_count = 0
        for i in range(2, num_labels):  # Ignorar el fondo (label 0) y el dado. Solo quedarse con los puntos.
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 20:  # Ajustar según el tamaño de los pips
                pip_count += 1
        # Agregar información del dado al diccionario
        dado = {
            "coords": (cx, cy),
            "roi": roi_cleaned,
            "pips": pip_count
        }
        dados_info.append(dado)
    return dados_info


def video_process(video):
    """
    Propósito: Procesar el video, detectar dados y devolver el frame donde los dados están quietos.
    Lógica:
        Carga el video y procesa cada frame.
        Detecta la máscara verde en el primer frame.
        Detecta dados rojos y verifica si hay exactamente 5 dados.
        Verifica si los dados están quietos usando la función stopped().
        Devuelve el primer frame donde los dados están quietos.
    Resultado: Devuelve:
        El frame original.
        La máscara de los dados.
        Las coordenadas de los dados."""
    i=0
    cap = cv2.VideoCapture(video)  # Capturar video desde un archivo:
    queue_coords = []            # Cola de coordenadas de los dados (últimos 5 frames)
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame)
        frame_number +=1
        # Detectar el paño verde
        if frame_number == 1: mask_green = create_green_mask(frame)
        # # Detectar los dados rojos dentro del paño verde
        mask_dados, dados_coords = detect_red_dados(frame, mask_green)
        i+=1
        print(i)
        if len(dados_coords) == 5:  # Solo analizamos si hay exactamente 5 dados detectados
            print(5)
            queue_coords.append(dados_coords)
            if len(queue_coords) > 5:  # Mantener la cola con un máximo de 5 elementos
                queue_coords.pop(0)
            # Verificar si los dados están quietos
            if len(queue_coords) == 5 and stopped(queue_coords):
                # imshow(frame, title='Frame detenido')
                cap.release()
                return frame, mask_dados, dados_coords
    cap.release()
    return None, None, None


def show_results(frame):
    """
    Muestra el frame original, la máscara verde, la máscara roja, 
    y la máscara de los dados (rojo limitado al verde) en subplots que comparten ejes.
    """
    # Crear las máscaras
    mask_green = create_green_mask(frame)
    mask_red = create_red_mask(frame)
    mask_dados = cv2.bitwise_and(mask_red, mask_green)
    # Crear los subplots con ejes compartidos
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    # Frame original
    axs[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convertir de BGR a RGB
    axs[0, 0].set_title("Frame Original")
    axs[0, 0].axis("off")
    # Máscara verde
    axs[0, 1].imshow(mask_green, cmap='gray')
    axs[0, 1].set_title("Máscara Verde")
    axs[0, 1].axis("off")
    # Máscara roja
    axs[1, 0].imshow(mask_red, cmap='gray')
    axs[1, 0].set_title("Máscara Roja")
    axs[1, 0].axis("off")
    # Máscara de dados (rojo limitado al verde)
    axs[1, 1].imshow(mask_dados, cmap='gray')
    axs[1, 1].set_title("Máscara Dados")
    axs[1, 1].axis("off")
    # Ajustar la disposición
    plt.tight_layout()
    #plt.show(block=False)
    plt.show()


def show_dados_info(frame, dados_info):
    """
    Muestra el frame original y los ROIs de los dados en subplots, incluyendo el número de pips.
    Args:
        frame (np.ndarray): Frame original.
        dados_info (list[dict]): Lista de diccionarios con información de los dados.
    """
    num_dados = len(dados_info)
    total_subplots = num_dados + 1  # Uno adicional para el frame original
    cols = 3  # Número de columnas en la cuadrícula
    rows = (total_subplots + cols - 1) // cols  # Calcular filas necesarias
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=False, sharey=False)
    axs = axs.ravel()  # Aplanar los ejes para indexarlos fácilmente
    # Mostrar el frame original en el primer subplot
    axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Frame Original")
    axs[0].axis("off")
    # Mostrar cada ROI de los dados en los subplots restantes
    for i, dado in enumerate(dados_info, start=1):
        coords = dado["coords"]
        pips = dado["pips"]
        roi = dado["roi"]
        axs[i].imshow(roi, cmap="gray")
        axs[i].set_title(f"Dado en {coords} - Pips: {pips}")
        axs[i].axis("off")
    # Deshabilitar los subplots sobrantes si hay menos dados que espacios
    for j in range(total_subplots, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    #plt.show(block=False)
    plt.show()



'''
Programa Principal
'''
# Parte a
frame, mask_dados, dados_coords = video_process('tirada_4.mp4')
show_results(frame)
# scores = get_dado_scores(mask_dados, dados_coords, roi_size=100)
dados_info = get_dados_info(mask_dados, dados_coords, roi_size=100)
# Llamar a la función para mostrar los ROIs de los dados
show_dados_info(frame, dados_info)


# Parte b
#--------- grabar video. Una modificación de la función process_video_ y usa todo lo demás

def video_record(input_video, output_video, roi_size=100):
    """Propósito: Generar un video procesado que resalte los dados y sus características.
    Lógica:
        Procesa el video cuadro a cuadro.
        Dibuja las ROI y las etiquetas con el número de pips para cada dado detectado.
        Guarda el video resultante.
    Resultado: Guarda un video donde los dados están identificados y etiquetados con el número de pips."""
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    queue_coords = []
    mask_green = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detectar la máscara del paño verde solo en el primer frame
        if mask_green is None:
            mask_green = create_green_mask(frame)
        # Detectar los dados
        mask_dados, dados_coords = detect_red_dados(frame, mask_green)
        # Si hay exactamente 5 dados, verificamos si están en reposo
        if len(dados_coords) == 5:
            queue_coords.append(dados_coords)
            if len(queue_coords) > 5:
                queue_coords.pop(0)
            if len(queue_coords) == 5 and stopped(queue_coords):
                # Calcular la información de los dados
                dados_info = get_dados_info(mask_dados, dados_coords, roi_size)
                # Dibujar bounding boxes y etiquetas
                for dado in dados_info:
                    cx, cy = dado["coords"]
                    pips = dado["pips"]
                    # Dibujar bounding box
                    x_start = max(cx - roi_size // 2, 0)
                    x_end = min(cx + roi_size // 2, frame.shape[1])
                    y_start = max(cy - roi_size // 2, 0)
                    y_end = min(cy + roi_size // 2, frame.shape[0])
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                    # Dibujar etiqueta con el número de pips
                    # label = f"Pips: {pips}"
                    label = f"{pips}"
                    cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 4)
        # Escribir el frame procesado en el archivo de salida
        out.write(frame)
        # Mostrar el frame (opcional, para debug)
        cv2.imshow('Processed Frame', cv2.resize(frame, (width // 3, height // 3)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Llamar a la función con el video de entrada y el nombre del video de salida
video_record('tirada_1.mp4', 'Video-Output-Processed1.mp4')

"""
Flujo principal del programa:
    Procesa el video (video_process) para detectar el frame donde los dados están quietos.
    Visualiza los resultados (show_results y show_dados_info).
    Procesa y graba un nuevo video con la información detallada de los dados (video_record).
Aspectos importantes:
    Segmentación por color: Usa el espacio de color HSV para segmentar colores, lo cual es más robusto que RGB en condiciones de iluminación variable.
    Análisis de movimiento: Usa una cola de posiciones recientes para verificar si los dados están quietos.
    Análisis morfológico: Limpia las máscaras y filtra ruido antes de extraer características como los pips."""

####################################################################################################################################################
                                                                    ## PRUEBAS - DESGLOSANDO CÓDIGO ##
####################################################################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

## FUNCION CUADRADOS ##################################################################################################################
# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
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


# Función para detectar contornos cuadrados
def detectar_contornos_cuadrados(frame):
    

    # Aplicar umbral adaptativo para resaltar contornos
    _, umbral = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos cuadrados aproximados
    
    contornos_cuadrados = [cnt for cnt in contornos if es_cuadrado(cnt) and cv2.contourArea(cnt) > 50*50]
    return contornos_cuadrados





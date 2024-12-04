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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Definir límites para el color verde en HSV (ajusta según el video)
    lower_green = np.array([35, 50, 50])  # H, S, V mínimos
    upper_green = np.array([85, 255, 255])  # H, S, V máximos
    # Crear máscara binaria del paño verde
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_fill = imfillhole(mask)
    return mask_fill


def create_red_mask(frame):
    """
    Detecta los dados rojos
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
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    # Limitar la detección de dados al área del paño verde
    mask_dados = cv2.bitwise_and(mask_red, mask_green)
    # Buscar contornos en la máscara resultante
    contours, _ = cv2.findContours(mask_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dados_coords = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filtrar ruido (ajusta el área mínima)
            # Obtener el centroide del contorno
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dados_coords.append((cx, cy))
    return mask_dados, dados_coords


def stopped(queue_coords, umbral=2):
    """
    Verifica si las coordenadas de los dados están quietas en la cola.
    """
    for i in range(1, len(queue_coords)):
        for (x1, y1), (x2, y2) in zip(queue_coords[i - 1], queue_coords[i]):
            if abs(x1 - x2) > umbral or abs(y1 - y2) > umbral:
                return False
    return True


def get_dados_info(mask_dados, dados_coords, roi_size=100):
    """
    Genera un diccionario con la información de cada dado detectado.
    Args:
        mask_dados (np.ndarray): Máscara binaria de los dados.
        dados_coords (list): Coordenadas de los centroides de los dados.
        roi_size (int): Tamaño de la ROI alrededor del dado.
    Returns:
        list[dict]: Lista de diccionarios con información de cada dado.
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
    i=0
    cap = cv2.VideoCapture(video)
    queue_coords = []  # Cola de coordenadas de los dados (últimos 5 frames)
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
    return None, None


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
    plt.show(block=False)


def show_dados_info(frame, dados_info):
    """
    Muestra el frame original y los ROIs de los dados en subplots.
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
    plt.show(block=False)



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
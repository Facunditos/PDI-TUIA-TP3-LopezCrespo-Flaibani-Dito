import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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


# --- Version 2 ------------------------------------------------
# Utilizando cv2.floodFill()
# SI rellena los huecos que tocan los bordes
def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
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
    mask_red = create_red_mask(frame)
    mask_dados= cv2.bitwise_and(mask_green, mask_red)
    kernel = np.ones((5, 5), np.uint8)
    mask_dados_fill = cv2.morphologyEx(mask_dados, cv2.MORPH_CLOSE, kernel)

    # Buscar contornos en la máscara resultante
    contours, _ = cv2.findContours(mask_dados_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calcula las áreas de todos los contornos
    areas = [cv2.moments(contour)["m00"] for contour in contours]
    # Calcula el área promedio
    if len(areas) > 0:  # Verifica que haya contornos
        area_promedio = sum(areas) / len(areas)
        print(f"El área promedio de los contornos es: {area_promedio} Nro. de componentes: {len(contours)}")
    else:
        print("No se encontraron contornos.")

    dados_coords = []
    for contour in contours:
        M = cv2.moments(contour)
        if (3500 <= M["m00"] <= 5500):  # Filtrar ruido (ajusta el área mínima)
            # Obtener el centroide del contorno
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dados_coords.append((cx, cy))
            print(f'Coords: {dados_coords} - Área: {M["m00"]}')
    return mask_dados_fill, dados_coords


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
        kernel = np.ones((5, 5), np.uint8)
        roi_cleaned = cv2.morphologyEx(roi_inverted, cv2.MORPH_OPEN, kernel)
        # Detectar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_cleaned, connectivity=8)
        # Filtrar componentes por área
        pip_count = 0
        for i in range(2, num_labels):  # Ignorar el fondo (label 0) y el dado. Solo quedarse con los puntos.
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 20:  # Ajustar según el tamaño de los pips
                pip_count += 1
        if 1 <= pip_count <= 6: # controla que el puntaje esté entre 1 y 6
            # Agregar información del dado al diccionario
            dado = {
                "coords": (cx, cy),
                "roi": roi_cleaned,
                "pips": pip_count
            }
            dados_info.append(dado)
    return dados_info


def video_process(video):
    i = 0
    flag_start = True
    frame_number = 0
    frame_number_start = -1
    frame_number_end = -1
    frame_end = None
    queue_coords = []  # Cola de coordenadas de los dados (últimos 5 frames)
    max_len_queue = 5
    cap = cv2.VideoCapture(video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame)
        # Detectar el paño verde
        if frame_number == 0: mask_green = create_green_mask(frame)
        # Detectar los dados rojos dentro del paño verde
        mask_dados, dados_coords = detect_red_dados(frame, mask_green)
        i+=1
        print(i)
        if len(dados_coords) == 5:  # Solo analizamos si hay exactamente 5 dados detectados
            print(5)
            queue_coords.append(dados_coords)
            if len(queue_coords) > max_len_queue:  # Mantener la cola con un máximo de 5 elementos
                queue_coords.pop(0)
            # Verificar si hay 5 dados y están quietos
            if len(queue_coords) == max_len_queue:
                # Está detenido
                if stopped(queue_coords):
                    # imshow(frame, title='Frame detenido')
                    if flag_start:
                        # Número del frame de inicio: detenido
                        frame_number_start = frame_number - max_len_queue + 1
                        frame_number_end = frame_number
                        frame_end = frame
                        mask_dados_end = mask_dados
                        dados_coords_end =dados_coords
                        flag_start = False
                    else:
                        # Número del frame final: detenido
                        frame_number_end = frame_number
                # Estuvo detenido y se volvió a mover
                elif not flag_start:
                    cap.release()
                    return frame_end, frame_number_start, frame_number_end, mask_dados_end, dados_coords_end
        # no hay 5 dados en la imagen
        elif not flag_start:
            cap.release()
            return frame_end, frame_number_start, frame_number_end, mask_dados_end, dados_coords_end
        frame_number +=1
    cap.release()
    if not flag_start:  # Si ya se habían detenido
        return frame_end, frame_number_start, frame_number_end, mask_dados_end, dados_coords_end

    # Si los dados nunca estuvieron quietos
    return None, -1, -1, None, None


def show_results(frame, mask_dados):
    """
    Muestra el frame original, la máscara verde, la máscara roja, 
    y la máscara de los dados (rojo limitado al verde) en subplots que comparten ejes.
    """
    # Crear las máscaras
    mask_green = create_green_mask(frame)
    mask_red = create_red_mask(frame)
    # mask_dados = cv2.bitwise_and(mask_red, mask_green)
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


def evaluar_generala(dados_info):
    # Extraer los valores (pips) de los dados
    valores_dados = [dado["pips"] for dado in dados_info]
    # Contar las ocurrencias de cada valor
    conteo = Counter(valores_dados)
    # Ordenar las ocurrencias en una lista [(valor, cantidad), ...]
    conteo_ordenado = conteo.most_common()
    # Verificar combinaciones del juego
    if len(conteo) == 1:  # Todos los dados tienen el mismo valor
        return "¡Generala!"
    elif conteo_ordenado[0][1] == 4:  # Cuatro dados iguales
        return "¡Póker!"
    elif conteo_ordenado[0][1] == 3 and len(conteo) == 2:  # Tres y dos iguales
        return "¡Full!"
    elif len(conteo) == 5 and sorted(valores_dados) in [list(range(1, 6)), list(range(2, 7))]:  # Escalera
        return "¡Escalera!"
    else:
        return "No se logró una combinación especial."


def video_record(input_video, output_video, frame_number_start, frame_number_end, dados_info, roi_size=100):
    frame_number = 0
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number_start < frame_number <= frame_number_end:
            for i, dado in enumerate(dados_info):
                cx, cy = dado["coords"]
                pips = dado["pips"]
                # Dibujar bounding box
                x_start = max(cx - roi_size // 2, 0)
                x_end = min(cx + roi_size // 2, frame.shape[1])
                y_start = max( cy - roi_size // 2, 0)
                y_end = min(cy + roi_size // 2, frame.shape[0])
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                # Dibujar etiqueta con el número de pips
                # label = f"Pips: {pips}"
                label = f"Dado {i+1}: {pips}"
                cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Escribir el frame procesado en el archivo de salida
        out.write(frame)
        # Mostrar el frame (opcional, para debug)
        cv2.imshow('Processed Frame', cv2.resize(frame, (width // 3, height // 3)))
        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()



'''
Programa Principal
'''
'''
ITEM A
'''
# Crear un diccionario para almacenar los datos de los videos
videos_info = {}
# Definir el directorio donde están los videos
video_dir = "videos"

# Iterar sobre los archivos en la carpeta "videos"
for video_file in os.listdir(video_dir):  # Lista los archivos en la carpeta "videos"
    if video_file.endswith(".mp4"):  # Filtrar solo archivos MP4
        video_path = os.path.join(video_dir, video_file)  # Crear la ruta completa
        print(video_path)
        print(f"Procesando video: {video_path}")
        # Procesar el video
        frame, frame_number_start, frame_number_end, mask_dados, dados_coords = video_process(video_path)
        # Mostrar resultados del procesamiento
        # imshow(frame)
        # show_results(frame, mask_dados)
        # Obtener información de los dados
        dados_info = get_dados_info(mask_dados, dados_coords, roi_size=100)
        # Mostrar información de los dados
        # show_dados_info(frame, dados_info)
        # Evaluar el resultado del juego
        game_result = evaluar_generala(dados_info)
        print(f"Resultado del juego para {video_file}: {game_result}")
        # Guardar la información en el diccionario
        videos_info[video_file] = {
            "frame_number_start": frame_number_start,
            "frame_number_end": frame_number_end,
            "dados_info": dados_info,
            "game_result": game_result,
        }

# Mostrar el diccionario completo
print("Información de los videos procesados:")
for video, info in videos_info.items():
    print(f"{video}: {info}")

'''
ITEM B
'''
# Definir la carpeta para guardar los videos procesados
output_dir = "videos_procesados"
# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre los datos de videos_info para grabar un nuevo video para cada entrada
for video_file, info in videos_info.items():
    input_video = os.path.join(video_dir, video_file)  # Ruta completa del video original
    output_video = os.path.join(output_dir, f"Processed_{video_file}")  # Ruta completa del archivo de salida

    # Extraer información necesaria
    frame_number_start = info["frame_number_start"]
    frame_number_end = info["frame_number_end"]
    dados_info = info["dados_info"]

    # Llamar a la función para procesar y grabar el nuevo video
    print(f"Grabando nuevo video: {output_video}")
    video_record(
        input_video,
        output_video,
        frame_number_start,
        frame_number_end,
        dados_info,
        roi_size=100
    )


#---------------------
#Código para pruebas y mostrar gráficos
# Item A
video = 'tirada_2.mp4'
frame, frame_number_start, frame_number_end, mask_dados, dados_coords = video_process(video)
imshow(frame)
show_results(frame, mask_dados)
dados_info = get_dados_info(mask_dados, dados_coords, roi_size=100)
# Llamar a la función para mostrar los ROIs de los dados
show_dados_info(frame, dados_info)

resultado = evaluar_generala(dados_info)
print(f"Resultado del juego: {resultado}")

# Item B
# Llamar a la función con el video de entrada y el nombre del video de salida
video_record(video, 'Video-Output-Processed_2.mp4', frame_number_start, frame_number_end, dados_info, roi_size=100)



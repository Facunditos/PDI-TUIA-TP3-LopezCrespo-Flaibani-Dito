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

# Función para determinar si un dado está quieto de un frame a otro
def dado_quieto(contornos_actual, contornos_anterior):
    
    for x,y in zip(contornos_actual,contornos_anterior):
        if np.array_equal(x, y):continue
        else: return False
    return True


def programa_dados(path):
    video_path = './' + path
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    f = 0
    contornos_anteriores = None
    intervalo_comparacion = 7  # Realizar la comparación cada 7 frames

    if not ret:
        print("No se pudo abrir el video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_procesado, frame_color = procesar_color(frame)
        contornos_cuadrados = detectar_contornos_cuadrados(frame_procesado)

        if len(contornos_cuadrados) == 5:
            if f % intervalo_comparacion == 0:  
                if contornos_anteriores is not None:
                    area_actual = sum(cv2.contourArea(contorno) for contorno in contornos_cuadrados)
                    area_anterior = sum(cv2.contourArea(contorno) for contorno in contornos_anteriores)

                    if abs(area_actual - area_anterior) < 100:  
                        #recortes = recortar_contornos(frame_procesado, contornos_cuadrados)
                        recortes = recortarxcontorno(frame_procesado, contornos_cuadrados)
                        #cv2.imshow('Frame Original', redimensionar(frame))
                        #cv2.imshow('Frame Original', redimensionar(frame_color))
                        #cv2.imshow('Frame Procesado', redimensionar(frame_procesado))

                        for i, recorte in enumerate(recortes):
                            valor = contarDados(recorte)

                            # Obtener las coordenadas del contorno cuadrado
                            x, y, w, h = cv2.boundingRect(contornos_cuadrados[i])

                            # Dibujar un bounding box en el frame original
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Mostrar el valor en el bounding box
                            cv2.putText(frame, f'Puntos: {valor}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        #cv2.imshow('Dados quietos con Lectura de cara', redimensionar(frame))
                        cv2.imwrite(f'./{path[:-4]}.jpg', redimensionar(frame))
            #print(f'{path} procesado con éxito')
                        #cv2.imshow('Frame Procesado', redimensionar(frame_procesado))

                contornos_anteriores = contornos_cuadrados

        f += 1

    #     if cv2.waitKey() & 0xFF == ord('q'):
    #         break

    # cap.release()
    #cv2.destroyAllWindows()


def grabar_video(path):
    video_path = './' + path
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir el video.")
        exit()

    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = video_path[2:-4]+'_procesado.mp4'
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    f = 0
    contornos_anteriores = None
    intervalo_comparacion = 7  # Realizar la comparación cada 7 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_procesado, _ = procesar_color(frame)
        contornos_cuadrados = detectar_contornos_cuadrados(frame_procesado)

        if len(contornos_cuadrados) > 0:
            if f % intervalo_comparacion == 0:
                if contornos_anteriores is not None:
                    area_actual = sum(cv2.contourArea(contorno) for contorno in contornos_cuadrados)
                    area_anterior = sum(cv2.contourArea(contorno) for contorno in contornos_anteriores)

                    if abs(area_actual - area_anterior) < 100:
                        recortes = recortarxcontorno(frame_procesado, contornos_cuadrados)

                        for i, recorte in enumerate(recortes):
                            valor = contarDados(recorte)
                            x, y, w, h = cv2.boundingRect(contornos_cuadrados[i])

                            # Dibujar un bounding box en el frame original
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            # Mostrar el valor en el bounding box
                            cv2.putText(frame, f'Puntos: {valor}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                contornos_anteriores = contornos_cuadrados

        out.write(frame)  
        #cv2.imshow('Video de Salida', frame)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    #cv2.destroyAllWindows() 













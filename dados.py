from os import makedirs,listdir,path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 

# Variables globales consumidas por las funciones
dir_videos_entrada = './videos_entradas'
videos_entradas = [vid for vid in listdir(dir_videos_entrada)]
dir_frames = './frames'
dir_videos_salida = './videos_salida'

# Funciones
def leer_video(video:str)->None:
    """
    Lee los videos disponibles en la carpeta y guarda sus respectivos frames 
    """    
    ruta_file_vid = path.join(dir_videos_entrada,video)
    file_vid_list = video.split('.')
    file_vid_sin_ext = file_vid_list[0]
    dir_frames_name = path.join(dir_frames,file_vid_sin_ext)
    makedirs(dir_frames_name, exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.
    # --- Leer un video ------------------------------------------------
    cap = cv2.VideoCapture(ruta_file_vid)  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
    # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
    frame_number = 0
    while (cap.isOpened()): # Verifica si el video se abrió correctamente.
        ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.
        if ret == True:  
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.
            #cv2.imshow('Frame', frame) # Muestra el frame redimensionado.
            ruta_frame = path.join(dir_frames_name, f"frame_{frame_number}.jpg")
            cv2.imwrite(ruta_frame, frame) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.
            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  
    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.

def grabar_videos(nombre_video:str,info_frames:dict):
    makedirs(dir_videos_salida, exist_ok = True)
    ruta_file_vid_entrada = path.join(dir_videos_entrada,nombre_video)
    ruta_file_vid_salida = path.join(dir_videos_salida,nombre_video)
    # --- Leer y grabar un video ------------------------------------------------
    cap = cv2.VideoCapture(ruta_file_vid_entrada)  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
    # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
    out = cv2.VideoWriter(ruta_file_vid_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    # Crear un objeto para escribir el video de salida.
    #   - 'Video-Output.mp4': Nombre del archivo de salida.
    #   - cv2.VideoWriter_fourcc(*'mp4v'): Codec utilizado para el archivo de salida.
    #   - fps: Cuadros por segundo del video de salida, debe coincidir con el video de entrada.
    #   - (width, height): Dimensiones del frame de salida, deben coincidir con las dimensiones originales del video.
    contador_frame = 0
    while (cap.isOpened()): # Verifica si el video se abrió correctamente.
        ret, frame = cap.read()  # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.    
        if ret == True:
            if (contador_frame in info_frames):
                for datos_caja in info_frames[contador_frame]:
                    p1,p2,etiqueta = datos_caja
                    p1_adaptado = tuple([coor*3 for coor in p1])
                    p2_adaptado = tuple([coor*3 for coor in p2])
                    cv2.rectangle(frame, p1_adaptado, p2_adaptado, (255,255,0), 4) 
            frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.
            cv2.imshow('Frame', frame_show) # Muestra el frame redimensionado.
            out.write(frame)   # Escribe el frame original (sin redimensionar) en el archivo de salida 'Video-Output.mp4'. IMPORTANTE: El tamaño del frame debe coincidir con el tamaño especificado al crear 'out'.
            contador_frame +=1
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:
            break
    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    out.release() # Libera el objeto 'out',  cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.

def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh



# Definimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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

def marcar_dados(frame,img,stats_dados):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    plt.title(f'{frame}')
    # Dibujar los rectángulos sobre los dados
    for stat_dado in stats_dados:
        x = stat_dado[0]
        y = stat_dado[1]
        ancho = stat_dado[2]
        alto = stat_dado[3]
        valor = 'dado'
        rect = Rectangle((x, y), ancho, alto,edgecolor=(255/255, 255/255, 0), linewidth=1, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 6, str(valor), fontsize=6, weight='bold')
    ax.axis('off')
    plt.show(block=False)

def obtener_canales_img_hsv(ruta_frame:str)->np.array:
    img = cv2.imread(ruta_frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
    h, s, v = cv2.split(img_hsv)
    """ plt.figure()
    ax1=plt.subplot(221); plt.imshow(img)
    plt.title(f'{ruta_frame}')
    plt.subplot(222, sharex=ax1, sharey=ax1), plt.imshow(h, cmap='gray'), plt.title('Canal H')
    plt.subplot(223, sharex=ax1, sharey=ax1), plt.imshow(s, cmap='gray'), plt.title('Canal S')
    plt.subplot(224, sharex=ax1, sharey=ax1), plt.imshow(v, cmap='gray'), plt.title('Canal V')
    plt.show(block=False) """
    return  h,s,v

def determinar_mascara_verde(ruta_primer_frame:str)->np.array:
    h,s,v = obtener_canales_img_hsv(ruta_primer_frame)
    # Segmentacion en color - Detectar el verde
    ix_h = np.logical_and(h > 180 * .4, h < 180 * .5)
    ix_s = np.logical_and(s > 256 * 0.6, s < 256)
    mascara_verde = np.logical_and(ix_h, ix_s)
    return mascara_verde


def determinar_mascara_roja(ruta_frame:str)->np.array:
    h,s,v = obtener_canales_img_hsv(ruta_frame)
    ix_h1 = np.logical_and(h > 180 * .9, h < 180) # ¿Por qué no alcanza con definir un límite inferior
    ix_h2 = h < 180 * 0.04 
    ix_s = np.logical_and(s > 256 * 0.46, s < 256) # Se utiliza este canal para excluir la mano
    mascara_roja = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)
    return mascara_roja

def detectar_dados(video:str):
    info_frames = {}
    file_vid_list = video.split('.')
    file_vid_sin_ext = file_vid_list[0]
    dir_frames_entrada = path.join(dir_frames,file_vid_sin_ext)
    frames = [frame for frame in listdir(dir_frames_entrada)]     
    primer_frame = frames[0]# Utilizo el primer frame para determinar la máscara verde
    path_primer_frame = path.join(dir_frames_entrada,primer_frame)
    mascara_verde = determinar_mascara_verde(path_primer_frame)
    coor_dados_quietos = None # cada elemento guarda las coordenadas de 5 objetos, existe igualdad en estas coordenadas
    q_frames_dados_quietos = 1
    stats_dados_quietos = None
    for ix_f,frame in enumerate(frames): # Esta lista de frames excluye al primero de ellos, utilizado para la máscara verde
        path_frame = path.join(dir_frames_entrada,frame)
        mascara_roja = determinar_mascara_roja(path_frame)
        mascara_elementos = np.logical_and(mascara_verde,mascara_roja).astype('uint8')
        #mascara_elementos_fh = imfillhole(mascara_elementos)
        # ----- Componentes 8 conectadas ---------
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_elementos, connectivity, cv2.CV_32S)
        for ix_l in range(1,num_labels):
            estadistica = stats[ix_l,:]
            x = estadistica[0]
            y = estadistica[1]
            ancho = estadistica[2]
            alto = estadistica[3]
            area_obj = estadistica[4]
            img_obj = mascara_elementos[y:y+alto,x:x+ancho]
            contours, hierarchy = cv2.findContours(img_obj, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            if area>15 and area<30:
                punto_1 = (x,y)
                punto_2 = (x+ancho,y+alto)
                caja = [punto_1,punto_2,None]
                if ix_f not in info_frames:
                    info_frames[ix_f] = [caja]
                else:
                    info_frames[ix_f].append(caja)
    return info_frames
            
                    perimeter = cv2.arcLength(cnt,True)
                    f_p = round(area / (perimeter**2),4) if perimeter != 0 else 0
                    if ix_f>64 and ix_f<100:
                        f_p
                    if (f_p>0.0697 and f_p<0.0740):
                        x_adaptado = x*3
                        y_adaptado = y*3
                        punto_1 = (x,y)
                        punto_2 = (x+ancho,y+alto)
                        caja = [punto_1,punto_2,f_p]
                        if ix_f not in info_frames:
                            info_frames[ix_f] =[caja]
                        else:
                            info_frames[ix_f].append(caja)
        
                    """ if (num_labels!=6): 
                        continue
                    else: # Interesan analizar aquellas máscaras que reconocen exactamente 6 elementos (fondo + 5 objetos)
                        stats_objetos =stats[1:,:] # Se excluye el fondo
                        coor_objs_frame_act = []
                        for stats_obj in stats_objetos:
                            esq_sup_izq = (stats_obj[0],stats_obj[1])
                            coor_objs_frame_act.append(esq_sup_izq)
                        coor_objs_frame_anterior = coor_dados_quietos
                        if q_frames_dados_quietos==3: # Se asume que si en 4 frames hay coincidencia de coor se correspoden con los dados quietos
                            marcar_dados(frame,img,stats_dados_quietos)
                            break
                        elif coor_objs_frame_anterior==coor_objs_frame_act:
                            q_frames_dados_quietos+=1
                        else:
                            coor_dados_quietos = coor_objs_frame_act
                            stats_dados_quietos = stats_objetos
                            q_frames_dados_quietos = 1
                """

for video in videos_entradas:
    #leer_video(video)
    datos_frames = detectar_dados(video)
    grabar_videos(video,datos_frames)



frames_vid_1_dir = './frames/tirada_1/'
# --- Espacio de color HSV ----------------------------------------------
# Utilizo el primer frame para determinar la máscara verde
img = cv2.imread(frames_vid_1_dir+'frame_68.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)
plt.figure()
ax1=plt.subplot(221); plt.imshow(img)
plt.subplot(222, sharex=ax1, sharey=ax1), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223, sharex=ax1, sharey=ax1), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.subplot(224, sharex=ax1, sharey=ax1), plt.imshow(v, cmap='gray'), plt.title('Canal V')
plt.show(block=False)



# Segmentacion en color - Detectar solo el ¿verde?
ix_h = np.logical_and(h > 180 * .4, h < 180 * .5)
ix_s = np.logical_and(s > 256 * 0.6, s < 256)
mascara_verde = np.logical_and(ix_h, ix_s)


# Analizo un frame en el cual estén presentes los dados
img = cv2.imread(frames_vid_1_dir+'frame_78.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)
plt.figure()
ax1=plt.subplot(221); plt.imshow(img)
plt.subplot(222, sharex=ax1, sharey=ax1), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223, sharex=ax1, sharey=ax1), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.subplot(224, sharex=ax1, sharey=ax1), plt.imshow(v, cmap='gray'), plt.title('Canal V')
plt.show(block=False)
# Segmentacion en color - Detectar solo el ¿rojo?
ix_h1 = np.logical_and(h > 180 * .9, h < 180) # ¿Por qué no alcanza con definir un límite inferior
ix_h2 = h < 180 * 0.04 # Sirve ser más exigente para descartar la mano 
ix_s = np.logical_and(s > 256 * 0.3, s < 256)
mascara_roja = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)

mascara_dado = np.logical_and(mascara_verde,mascara_roja).astype('uint8')
imshow(mascara_verde)
imshow(mascara_roja)
imshow(mascara_dado)

# ----- Componentes 8 conectadas ---------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_dado, connectivity, cv2.CV_32S)
esq_sup_izq_dados = []
stats_dados =stats[1:,:]
for dado in stats_dados:
    esq_sup_izq = (dado[0],dado[1])
    esq_sup_izq_dados.append(esq_sup_izq)

esq_sup_izq_dados    

['','b'] == ['a','b']

r, g, b = cv2.split(img)
r[mascara_dado != True] = 0
g[mascara_dado != True] = 0
b[mascara_dado != True] = 0
dado_img = cv2.merge((r, g, b))
plt.figure(), plt.imshow(dado_img), plt.show(block=False)



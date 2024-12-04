import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from matplotlib.patches import Rectangle 


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


for n in range(1,5):
    dir_frames_entrada = f'./frames_entrada/tirada_{n}/'
    frames = [frame for frame in listdir(dir_frames_entrada)]     
    # Utilizo el primer frame para determinar la máscara verde
    primer_frame = frames.pop(0)
    img = cv2.imread(dir_frames_entrada+primer_frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
    h, s, v = cv2.split(img_hsv)
    # Segmentacion en color - Detectar el verde
    ix_h = np.logical_and(h > 180 * .4, h < 180 * .5)
    ix_s = np.logical_and(s > 256 * 0.6, s < 256)
    mascara_verde = np.logical_and(ix_h, ix_s)
    coor_dados_quietos = None # cada elemento guarda las coordenadas de 5 objetos, existe igualdad en estas coordenadas
    q_frames_dados_quietos = 1
    stats_dados_quietos = None
    for frame in frames: # Esta lista de frames excluye al primero de ellos, utilizado para la máscara verde
        img = cv2.imread(dir_frames_entrada+frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
        h, s, v = cv2.split(img_hsv)
        ix_h1 = np.logical_and(h > 180 * .9, h < 180) # ¿Por qué no alcanza con definir un límite inferior
        ix_h2 = h < 180 * 0.04 
        ix_s = np.logical_and(s > 256 * 0.3, s < 256) # Se utiliza este canal para excluir la mano
        mascara_roja = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)
        mascara_dado = np.logical_and(mascara_verde,mascara_roja).astype('uint8')
        # ----- Componentes 8 conectadas ---------
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_dado, connectivity, cv2.CV_32S)
        if (num_labels!=6): 
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
            

frames_vid_1_dir = './frames_entrada/tirada_1/'
# --- Espacio de color HSV ----------------------------------------------
# Utilizo el primer frame para determinar la máscara verde
img = cv2.imread(dir_frames_entrada+'frame_88.jpg')
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



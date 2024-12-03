import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- Espacio de color HSV ----------------------------------------------
# Utilizo el primer frame para determinar la máscara verde
img = cv2.imread('./frames/frame_1.jpg')
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
img = cv2.imread('./frames/frame_79.jpg')
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
ix_h2 = h < 180 * 0.04
ix_s = np.logical_and(s > 256 * 0.3, s < 256)
mascara_roja = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)

mascara_dado = np.logical_and(mascara_verde,mascara_roja).astype('uint8')
imshow(mascara_verde)
imshow(mascara_roja)
imshow(mascara_dado)

# ----- Componentes 8 conectadas ---------
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_dado, connectivity, cv2.CV_32S)
punta_alta_dados = []
stats_dados =stats[1:,:]
for dado in stats_dados:
    esq_sup_izq = (dado[0],dado[1])
    punta_alta_dados.append(esq_sup_izq)

punta_alta_dados    


r, g, b = cv2.split(img)
r[mascara_dado != True] = 0
g[mascara_dado != True] = 0
b[mascara_dado != True] = 0
dado_img = cv2.merge((r, g, b))
plt.figure(), plt.imshow(dado_img), plt.show(block=False)
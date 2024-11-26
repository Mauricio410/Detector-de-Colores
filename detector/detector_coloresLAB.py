import cv2
import numpy as np

archivoVid = "1.avi"
archivoImg = "1.jpg"
archivoCam = 0

cap = cv2.VideoCapture(archivoVid)

def nothing(x):
    pass

    # LAB = Luminosidad/Lightness, A, B

cv2.namedWindow('Parametros')

cv2.createTrackbar('Luminosidad Min', 'Parametros', 0, 255, nothing)
cv2.createTrackbar('Luminosidad Max', 'Parametros', 0, 255, nothing) # 255 es el valor maximo de Luminosidad/Lightness

cv2.createTrackbar('A Min', 'Parametros', 0, 255, nothing)
cv2.createTrackbar('A Max', 'Parametros', 0, 255, nothing) # 255 es el valor maximo de A

cv2.createTrackbar('B Min', 'Parametros', 0, 255, nothing)
cv2.createTrackbar('B Max', 'Parametros', 0, 255, nothing) # 255 es el valor maximo de B

cv2.createTrackbar('Kernel X', 'Parametros', 1, 30, nothing) # Kernel para operaciones morfologicas
cv2.createTrackbar('Kernel Y', 'Parametros', 1, 30, nothing) # Kernel para operaciones morfologicas

paused = False

while (1):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        LMin = cv2.getTrackbarPos('Luminosidad Min', 'Parametros')
        LMax = cv2.getTrackbarPos('Luminosidad Max', 'Parametros')

        AMin = cv2.getTrackbarPos('A Min', 'Parametros')
        AMax = cv2.getTrackbarPos('A Max', 'Parametros')

        BMin = cv2.getTrackbarPos('B Min', 'Parametros')
        BMax = cv2.getTrackbarPos('B Max', 'Parametros')

        kX = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kY = cv2.getTrackbarPos('Kernel Y', 'Parametros')

        claros = np.array([LMin, AMin, BMin])
        oscuros = np.array([LMax, AMax, BMax])

        mask = cv2.inRange(lab, claros, oscuros)

        kernel = np.ones((kX, kY), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornos, -1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

    k = cv2.waitKey(25) & 0xFF
    if k == 27:  # Tecla 'Esc' para salir
        break
    elif k == ord('p'):  # Tecla 'p' para pausar/reanudar
        paused = not paused

cap.release()
cv2.destroyAllWindows()
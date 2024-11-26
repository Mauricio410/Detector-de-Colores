import numpy as np
import cv2

archivoVid = "videos/video_sombraOK.avi"
archivoCam = "fotos/imagen1.jpg"

def nada (x):
    pass

cv2.namedWindow('Parametros')

cv2.createTrackbar('Tonalidad Minimo', 'Parametros', 0, 179, nada)
cv2.createTrackbar('Tonalidad Maximo', 'Parametros', 0, 179, nada)
cv2.createTrackbar('Pureza Minimo', 'Parametros', 0, 255, nada)
cv2.createTrackbar('Pureza Maximo', 'Parametros', 0, 255, nada)
cv2.createTrackbar('Luminosidad Minimo', 'Parametros', 0, 255, nada)
cv2.createTrackbar('Luminosidad Maximo', 'Parametros', 0, 255, nada)
cv2.createTrackbar('Kernel X', 'Parametros', 1, 30, nada)
cv2.createTrackbar('Kernel Y', 'Parametros', 1, 30, nada)

cap = cv2.VideoCapture(archivoVid)

paused = False

while (1):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        TMin = cv2.getTrackbarPos('Tonalidad Minimo', 'Parametros')
        TMax = cv2.getTrackbarPos('Tonalidad Maximo', 'Parametros')
        PMin = cv2.getTrackbarPos('Pureza Minimo', 'Parametros')
        PMax = cv2.getTrackbarPos('Pureza Maximo', 'Parametros')
        LMin = cv2.getTrackbarPos('Luminosidad Minimo', 'Parametros')
        LMax = cv2.getTrackbarPos('Luminosidad Maximo', 'Parametros')
        kX = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kY = cv2.getTrackbarPos('Kernel Y', 'Parametros')

        color_oscuro = np.array([TMin, PMin, LMin])
        color_claro = np.array([TMax, PMax, LMax])

        mask = cv2.inRange(hsv, color_oscuro, color_claro)

        kernelx = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kernely = cv2.getTrackbarPos('Kernel Y', 'Parametros')

        kernel = np.ones((kernelx, kernely), np.uint8)
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
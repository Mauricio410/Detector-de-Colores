import cv2
import numpy as np

archivoVid = "video_sombraOK.avi"
archivoCam = "fotos/imagen1.jpg"

def nothing(x):
    pass

cv2.namedWindow('Parametros')

# HSL = Hue, Saturation, Lightness

cv2.namedWindow('Parametros')
cv2.createTrackbar('Y Min', 'Parametros', 110, 255, lambda x: None)
cv2.createTrackbar('Y Max', 'Parametros', 185, 255, lambda x: None)

cv2.createTrackbar('Cr Min', 'Parametros', 0, 255, lambda x: None)
cv2.createTrackbar('Cr Max', 'Parametros', 175, 255, lambda x: None)

cv2.createTrackbar('Cb Min', 'Parametros', 110, 255, lambda x: None)
cv2.createTrackbar('Cb Max', 'Parametros', 140, 255, lambda x: None)

cv2.createTrackbar('Kernel X', 'Parametros', 6, 30, lambda x: None)
cv2.createTrackbar('Kernel Y', 'Parametros', 6, 30, lambda x: None)


cap = cv2.VideoCapture(archivoVid)

paused = False

while (1):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break


        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        YMin = cv2.getTrackbarPos('Y Min', 'Parametros')
        YMax = cv2.getTrackbarPos('Y Max', 'Parametros')
        CrMin = cv2.getTrackbarPos('Cr Min', 'Parametros')
        CrMax = cv2.getTrackbarPos('Cr Max', 'Parametros')
        CbMin = cv2.getTrackbarPos('Cb Min', 'Parametros')
        CbMax = cv2.getTrackbarPos('Cb Max', 'Parametros')
        kX = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kY = cv2.getTrackbarPos('Kernel Y', 'Parametros')

        color_oscuro = np.array([YMin, CrMin, CbMin])
        color_claro = np.array([YMax, CrMax, CbMax])

        mask = cv2.inRange(ycrcb, color_oscuro, color_claro)
        
        kernel = np.ones((kX, kY), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornos, -1, (0, 0, 0), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

    k = cv2.waitKey(50) & 0xFF
    if k == 27:  # Tecla 'Esc' para salir
        break
    elif k == ord('p'):  # Tecla 'p' para pausar/reanudar
        paused = not paused

cap.release()
cv2.destroyAllWindows()
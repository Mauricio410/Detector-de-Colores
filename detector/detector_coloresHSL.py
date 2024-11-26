import cv2
import numpy as np

archivoVid = "video_sombraOK.avi"
archivoCam = "fotos/imagen1.jpg"

def nothing(x):
    pass

cv2.namedWindow('Parametros')

# HSL = Hue, Saturation, Lightness

cv2.namedWindow('Parametros')
cv2.createTrackbar('H Min', 'Parametros', 0, 179, lambda x: None)
cv2.createTrackbar('H Max', 'Parametros', 179, 179, lambda x: None)

cv2.createTrackbar('S Min', 'Parametros', 40, 255, lambda x: None)
cv2.createTrackbar('S Max', 'Parametros', 255, 255, lambda x: None)

cv2.createTrackbar('L Min', 'Parametros', 0, 255, lambda x: None)
cv2.createTrackbar('L Max', 'Parametros', 160, 255, lambda x: None)

cv2.createTrackbar('Kernel X', 'Parametros', 6, 30, lambda x: None)
cv2.createTrackbar('Kernel Y', 'Parametros', 6, 30, lambda x: None)


cap = cv2.VideoCapture(archivoVid)

paused = False

while (1):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        HMin = cv2.getTrackbarPos('H Min', 'Parametros')
        HMax = cv2.getTrackbarPos('H Max', 'Parametros')
        SMin = cv2.getTrackbarPos('S Min', 'Parametros')
        SMax = cv2.getTrackbarPos('S Max', 'Parametros')
        LMin = cv2.getTrackbarPos('L Min', 'Parametros')
        LMax = cv2.getTrackbarPos('L Max', 'Parametros')
        kX = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kY = cv2.getTrackbarPos('Kernel Y', 'Parametros')


        color_oscuro = np.array([HMin, LMin, SMin])
        color_claro = np.array([HMax, LMax, SMax])

        mask = cv2.inRange(hls, color_oscuro, color_claro)

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
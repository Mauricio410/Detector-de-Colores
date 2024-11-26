import cv2
import numpy as np

archivoVid = "video_sombraOK.avi"
archivoCam = "fotos/imagen1.jpg"

def nothing(x):
    pass

cv2.namedWindow('Parametros')

# HSL = Hue, Saturation, Lightness

cv2.namedWindow('Parametros')
cv2.createTrackbar('R Min', 'Parametros', 40, 255, nothing)
cv2.createTrackbar('R Max', 'Parametros', 130, 255, nothing)
cv2.createTrackbar('G Min', 'Parametros', 0, 255, nothing)
cv2.createTrackbar('G Max', 'Parametros', 255, 255, nothing)
cv2.createTrackbar('B Min', 'Parametros', 0, 255, nothing)
cv2.createTrackbar('B Max', 'Parametros', 255, 255, nothing)
cv2.createTrackbar('Kernel X', 'Parametros', 6, 30, nothing)
cv2.createTrackbar('Kernel Y', 'Parametros', 6, 30, nothing)

cap = cv2.VideoCapture(archivoVid)

paused = False

while (1):
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        RMin = cv2.getTrackbarPos('R Min', 'Parametros')
        RMax = cv2.getTrackbarPos('R Max', 'Parametros')
        GMin = cv2.getTrackbarPos('G Min', 'Parametros')
        GMax = cv2.getTrackbarPos('G Max', 'Parametros')
        BMin = cv2.getTrackbarPos('B Min', 'Parametros')
        BMax = cv2.getTrackbarPos('B Max', 'Parametros')
        kX = cv2.getTrackbarPos('Kernel X', 'Parametros')
        kY = cv2.getTrackbarPos('Kernel Y', 'Parametros')

        color_oscuro = np.array([BMin, GMin, RMin])
        color_claro = np.array([BMax, GMax, RMax])

        mask = cv2.inRange(frame, color_oscuro, color_claro)
        
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
import cv2
import numpy as np

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Función de callback para los trackbars
def nothing(x):
    pass

# Crea una ventana
cv2.namedWindow('Trackbars')

# Crea los trackbars
#cv2.createTrackbar('Autofocus', 'Trackbars', 1, 1, nothing)
#cv2.createTrackbar('Focus', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Brightness', 'Trackbars', 25, 255, nothing)
cv2.createTrackbar('Contrast', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('Saturation', 'Trackbars', 75, 255, nothing)
cv2.createTrackbar('Sharpness', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Gamma', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Hue', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Gain', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Backlight', 'Trackbars', 0, 1, nothing)
cv2.createTrackbar('Red Balance', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Blue Balance', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Green Balance', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Temperature', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Trigger', 'Trackbars', 0, 1, nothing)
cv2.createTrackbar('Monochrome', 'Trackbars', 0, 1, nothing)
cv2.createTrackbar('Exposure', 'Trackbars', 0, 1, nothing)
cv2.createTrackbar('Iris', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('White Balance', 'Trackbars', 0, 1, nothing)

# Rango de colores en HSV
lower_color = np.array([30, 150, 50])
upper_color = np.array([85, 255, 255])

while True:
    # Lee los valores de los trackbars
    #autofocus = cv2.getTrackbarPos('Autofocus', 'Trackbars')
    #focus = cv2.getTrackbarPos('Focus', 'Trackbars')
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars')
    contrast = cv2.getTrackbarPos('Contrast', 'Trackbars')
    saturation = cv2.getTrackbarPos('Saturation', 'Trackbars')
    sharpness = cv2.getTrackbarPos('Sharpness', 'Trackbars')
    gamma = cv2.getTrackbarPos('Gamma', 'Trackbars')
    hue = cv2.getTrackbarPos('Hue', 'Trackbars')
    gain = cv2.getTrackbarPos('Gain', 'Trackbars')
    backlight = cv2.getTrackbarPos('Backlight', 'Trackbars')
    redbalance = cv2.getTrackbarPos('Red Balance', 'Trackbars')
    bluebalance = cv2.getTrackbarPos('Blue Balance', 'Trackbars')
    temp = cv2.getTrackbarPos('Temperature', 'Trackbars')
    monochrome = cv2.getTrackbarPos('Monochrome', 'Trackbars')
    exposure = cv2.getTrackbarPos('Exposure', 'Trackbars')
    iris = cv2.getTrackbarPos('Iris', 'Trackbars')
    whitebalance = cv2.getTrackbarPos('White Balance', 'Trackbars')

    # Aplica los valores a la cámara
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
    #cap.set(cv2.CAP_PROP_FOCUS, focus)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
    cap.set(cv2.CAP_PROP_GAMMA, gamma)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_BACKLIGHT, backlight)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, redbalance)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, bluebalance)
    cap.set(cv2.CAP_PROP_TEMPERATURE, temp)
    cap.set(cv2.CAP_PROP_MONOCHROME, monochrome)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_IRIS, iris)
    cap.set(cv2.CAP_PROP_AUTO_WB, whitebalance)

    # Captura el frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte el frame a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crea una máscara con el rango de colores
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplica la máscara al frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Muestra el frame original y el resultado
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
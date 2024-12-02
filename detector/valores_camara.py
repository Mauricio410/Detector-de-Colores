import cv2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Crear una ventana para mostrar los valores
cv2.namedWindow('Camera Properties')

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener los valores de las propiedades de la cámara
    autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
    gamma = cap.get(cv2.CAP_PROP_GAMMA)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    backlight = cap.get(cv2.CAP_PROP_BACKLIGHT)
    redbalance = cap.get(cv2.CAP_PROP_AUTO_WB)
    bluebalance = cap.get(cv2.CAP_PROP_AUTO_WB)
    greenbalance = cap.get(cv2.CAP_PROP_AUTO_WB)
    temperature = cap.get(cv2.CAP_PROP_TEMPERATURE)
    trigger = cap.get(cv2.CAP_PROP_TRIGGER)
    monochrome = cap.get(cv2.CAP_PROP_MONOCHROME)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    iris = cap.get(cv2.CAP_PROP_IRIS)
    white_balance = cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)

    # Mostrar los valores en la ventana
    info = f'Autofocus: {autofocus}\nFocus: {focus}\nBrightness: {brightness}\nContrast: {contrast}\nSaturation: {saturation}\nSharpness: {sharpness}\nGamma: {gamma}\nHue: {hue}\nGain: {gain}\nBacklight: {backlight}\nRed Balance: {redbalance}\nBlue Balance: {bluebalance}\nGreen Balance: {greenbalance}\nTemperature: {temperature}\nTrigger: {trigger}\nMonochrome: {monochrome}\nExposure: {exposure}\nIris: {iris}\nWhite Balance: {white_balance}'
    y0, dy = 20, 20
    for i, line in enumerate(info.split('\n')):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Mostrar el frame con los valores
    cv2.imshow('Camera Properties', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
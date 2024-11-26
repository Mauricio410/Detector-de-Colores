import numpy as np
import cv2

def nothing(x):
    pass

# Crear una ventana
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 400)  # Ajusta el tamaño de la ventana según tus necesidades

# Crear trackbars para ajustar los valores HSV
# HSV (Hue, Saturation, Value)

cv2.createTrackbar('H Lower Claros', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Upper Claros', 'Trackbars', 50, 179, nothing)
cv2.createTrackbar('S Lower Claros', 'Trackbars', 80, 255, nothing)
cv2.createTrackbar('S Upper Claros', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Lower Claros', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('V Upper Claros', 'Trackbars', 210, 255, nothing)

cv2.createTrackbar('H Lower Oscuros', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Upper Oscuros', 'Trackbars', 18, 179, nothing)
cv2.createTrackbar('S Lower Oscuros', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('S Upper Oscuros', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Lower Oscuros', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Upper Oscuros', 'Trackbars', 80, 255, nothing)

cv2.createTrackbar('Kernel x', 'Trackbars', 2, 20, nothing)
cv2.createTrackbar('Kernel y', 'Trackbars', 2, 20, nothing)

archivo = cv2.VideoCapture('1.avi')

paused = False

while(1):
    if not paused:
        _, imageFrame = archivo.read()

        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Leer los valores de las trackbars
        h_lower_claro = cv2.getTrackbarPos('H Lower Claros', 'Trackbars')
        s_lower_claro = cv2.getTrackbarPos('S Lower Claros', 'Trackbars')
        v_lower_claro = cv2.getTrackbarPos('V Lower Claros', 'Trackbars')

        h_upper_claro = cv2.getTrackbarPos('H Upper Claros', 'Trackbars')
        s_upper_claro = cv2.getTrackbarPos('S Upper Claros', 'Trackbars')
        v_upper_claro = cv2.getTrackbarPos('V Upper Claros', 'Trackbars')

        # Definir los rangos de color usando los valores de las trackbars
        amarillo_claro_lower = np.array([h_lower_claro, s_lower_claro, v_lower_claro], np.uint8)
        amarillo_claro_upper = np.array([h_upper_claro, s_upper_claro, v_upper_claro], np.uint8)
        amarillo_claro_mask = cv2.inRange(hsvFrame, amarillo_claro_lower, amarillo_claro_upper)

        h_lower_oscuro = cv2.getTrackbarPos('H Lower Oscuros', 'Trackbars')
        s_lower_oscuro = cv2.getTrackbarPos('S Lower Oscuros', 'Trackbars')
        v_lower_oscuro = cv2.getTrackbarPos('V Lower Oscuros', 'Trackbars')

        h_upper_oscuro = cv2.getTrackbarPos('H Upper Oscuros', 'Trackbars')
        s_upper_oscuro = cv2.getTrackbarPos('S Upper Oscuros', 'Trackbars')
        v_upper_oscuro = cv2.getTrackbarPos('V Upper Oscuros', 'Trackbars')

        amarillo_oscuro_lower = np.array([h_lower_oscuro, s_lower_oscuro, v_lower_oscuro], np.uint8)
        amarillo_oscuro_upper = np.array([h_upper_oscuro, s_upper_oscuro, v_upper_oscuro], np.uint8)
        amarillo_oscuro_mask = cv2.inRange(hsvFrame, amarillo_oscuro_lower, amarillo_oscuro_upper)

        # Morphological Transform, Dilation
        kernel_x = cv2.getTrackbarPos('Kernel x', 'Trackbars')
        kernel_y = cv2.getTrackbarPos('Kernel y', 'Trackbars')
        kernel = np.ones((kernel_x, kernel_y), "uint8")

        # Amariilo claro
        amarillo_claro_mask = cv2.dilate(amarillo_claro_mask, kernel)
        res_amarillo_claro = cv2.bitwise_and(imageFrame, imageFrame, mask=amarillo_claro_mask)

        # Amarillo oscuro
        amarillo_oscuro_mask = cv2.dilate(amarillo_oscuro_mask, kernel)
        res_amarillo_oscuro = cv2.bitwise_and(imageFrame, imageFrame, mask=amarillo_oscuro_mask)

        # Contorno Amarillo Claro
        contours, hierarchy = cv2.findContours(amarillo_claro_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.putText(imageFrame, "Amarillo Claro", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        # Contorno Amarillo Oscuro
        contours, hierarchy = cv2.findContours(amarillo_oscuro_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(imageFrame, "Amarillo Oscuro", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # Combinar las mask amarillo claro y oscuro
    mask = cv2.bitwise_or(amarillo_claro_mask, amarillo_oscuro_mask)

    # Mostrar las imágenes
    cv2.imshow("Detector de color", imageFrame)
    cv2.imshow("Amarillo Claro Mask", amarillo_claro_mask)
    cv2.imshow("Amarillo Oscuro Mask", amarillo_oscuro_mask)
    cv2.imshow("Combinada", mask)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        archivo.release()
        cv2.destroyAllWindows()
        break
    elif key == ord('p'):
        paused = not paused
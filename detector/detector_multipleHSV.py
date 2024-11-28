import numpy as np
import cv2

def nothing(x):
    pass

# Crear una ventana
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 600, 400)  # Ajusta el tamaño de la ventana según tus necesidades

# Crear trackbars para ajustar los valores HSV
# HSV (Hue, Saturation, Value)

cv2.createTrackbar('H Lower', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Upper', 'Trackbars', 50, 179, nothing)
cv2.createTrackbar('S Lower', 'Trackbars', 80, 255, nothing)
cv2.createTrackbar('S Upper', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('V Upper', 'Trackbars', 210, 255, nothing)

cv2.createTrackbar('H Lower Oscuros', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Upper Oscuros', 'Trackbars', 18, 179, nothing)
cv2.createTrackbar('S Lower Oscuros', 'Trackbars', 20, 255, nothing)
cv2.createTrackbar('S Upper Oscuros', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Lower Oscuros', 'Trackbars', 60, 255, nothing)
cv2.createTrackbar('V Upper Oscuros', 'Trackbars', 80, 255, nothing)

cv2.createTrackbar('H Lower Claros', 'Trackbars', 20, 179, nothing)
cv2.createTrackbar('H Upper Claros', 'Trackbars', 65, 179, nothing)
cv2.createTrackbar('S Lower Claros', 'Trackbars', 60, 255, nothing)
cv2.createTrackbar('S Upper Claros', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Lower Claros', 'Trackbars', 185, 255, nothing)
cv2.createTrackbar('V Upper Claros', 'Trackbars', 255, 255, nothing)


cv2.createTrackbar('Kernel x', 'Trackbars', 2, 20, nothing)
cv2.createTrackbar('Kernel y', 'Trackbars', 2, 20, nothing)

archivo = cv2.VideoCapture('1.avi')

paused = False

while(1):
    if not paused:
        _, imageFrame = archivo.read()

        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Leer los valores de las trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'Trackbars')
        s_lower = cv2.getTrackbarPos('S Lower', 'Trackbars')
        v_lower = cv2.getTrackbarPos('V Lower', 'Trackbars')

        h_upper = cv2.getTrackbarPos('H Upper', 'Trackbars')
        s_upper = cv2.getTrackbarPos('S Upper', 'Trackbars')
        v_upper = cv2.getTrackbarPos('V Upper', 'Trackbars')

        # Definir los rangos de color usando los valores de las trackbars
        color_lower = np.array([h_lower, s_lower, v_lower], np.uint8)
        color_upper = np.array([h_upper, s_upper, v_upper], np.uint8)
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper)

        # Leer los valores de las trackbars
        h_lower_oscuro = cv2.getTrackbarPos('H Lower Oscuros', 'Trackbars')
        s_lower_oscuro = cv2.getTrackbarPos('S Lower Oscuros', 'Trackbars')
        v_lower_oscuro = cv2.getTrackbarPos('V Lower Oscuros', 'Trackbars')

        h_upper_oscuro = cv2.getTrackbarPos('H Upper Oscuros', 'Trackbars')
        s_upper_oscuro = cv2.getTrackbarPos('S Upper Oscuros', 'Trackbars')
        v_upper_oscuro = cv2.getTrackbarPos('V Upper Oscuros', 'Trackbars')

        # Definir los rangos de color usando los valores de las trackbars
        color_oscuro_lower = np.array([h_lower_oscuro, s_lower_oscuro, v_lower_oscuro], np.uint8)
        color_oscuro_upper = np.array([h_upper_oscuro, s_upper_oscuro, v_upper_oscuro], np.uint8)
        color_oscuro_mask = cv2.inRange(hsvFrame, color_oscuro_lower, color_oscuro_upper)

        # Leer los valores de las trackbars
        h_lower_claro = cv2.getTrackbarPos('H Lower Claros', 'Trackbars')
        s_lower_claro = cv2.getTrackbarPos('S Lower Claros', 'Trackbars')
        v_lower_claro = cv2.getTrackbarPos('V Lower Claros', 'Trackbars')

        h_upper_claro = cv2.getTrackbarPos('H Upper Claros', 'Trackbars')
        s_upper_claro = cv2.getTrackbarPos('S Upper Claros', 'Trackbars')
        v_upper_claro = cv2.getTrackbarPos('V Upper Claros', 'Trackbars')

        # Definir los rangos de color usando los valores de las trackbars
        color_claro_lower = np.array([h_lower_claro, s_lower_claro, v_lower_claro], np.uint8)
        color_claro_upper = np.array([h_upper_claro, s_upper_claro, v_upper_claro], np.uint8)
        color_claro_mask = cv2.inRange(hsvFrame, color_claro_lower, color_claro_upper)

        # Morphological Transform, Dilation
        kernel_x = cv2.getTrackbarPos('Kernel x', 'Trackbars')
        kernel_y = cv2.getTrackbarPos('Kernel y', 'Trackbars')
        kernel = np.ones((kernel_x, kernel_y), "uint8")

        # Amariilo claro
        color_mask = cv2.dilate(color_mask, kernel)
        res_color = cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)

        # Amarillo oscuro
        color_oscuro_mask = cv2.dilate(color_oscuro_mask, kernel)
        res_color_oscuro = cv2.bitwise_and(imageFrame, imageFrame, mask=color_oscuro_mask)

        # Amarillo luz
        color_claro_mask = cv2.dilate(color_claro_mask, kernel)
        res_color_claro = cv2.bitwise_and(imageFrame, imageFrame, mask=color_claro_mask)

        # Contorno Amarillo Claro
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.putText(imageFrame, "Color neutro", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        # Contorno Amarillo Oscuro
        contours, hierarchy = cv2.findContours(color_oscuro_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(imageFrame, "Color oscuro", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        # Contorno Amarillo Luz
        contours, hierarchy = cv2.findContours(color_claro_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(imageFrame, "Color claro", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
    # Combinar las mask amarillo claro y oscuro
    mask = cv2.bitwise_or(color_mask, color_oscuro_mask, color_claro_mask)

    # Mostrar las imágenes
    cv2.imshow("Detector de color", imageFrame)
    cv2.imshow("Color Mask", color_mask)
    cv2.imshow("Color Oscuro Mask", color_oscuro_mask)
    cv2.imshow("Color Claro Mask", color_claro_mask)
    cv2.imshow("Combinada", mask)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        archivo.release()
        cv2.destroyAllWindows()
        break
    elif key == ord('p'):
        paused = not paused
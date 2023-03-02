import cv2

# Carica l'immagine
image = cv2.imread(r"public\ball.png")

# Crea una finestra per mostrare l'immagine
cv2.namedWindow("Immagine")

# Definisci la funzione per gestire i clic del mouse
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Coordinate: ({x}, {y})")

# Associa la funzione alla finestra
cv2.setMouseCallback("Immagine", mouse_callback)

# Mostra l'immagine
cv2.imshow("Immagine", image)

# Attendi il clic del mouse
cv2.waitKey(0)

# Chiudi la finestra
cv2.destroyAllWindows()

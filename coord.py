import cv2

# Carica l'immagine
video = cv2.VideoCapture("public/test.mp4")

# Leggi il primo frame
ret, frame = video.read()

# Crea una finestra per mostrare l'immagine
cv2.namedWindow("Immagine")

# Definisci la funzione per gestire i clic del mouse
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Coordinate: ({x}, {y})")

# Associa la funzione alla finestra
cv2.setMouseCallback("Immagine", mouse_callback)

# Mostra l'immagine
cv2.imshow("Immagine", frame)

# Attendi il clic del mouse
video.release()
cv2.waitKey(0)

# Chiudi la finestra
cv2.destroyAllWindows()

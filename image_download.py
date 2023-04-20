import cv2
import os

# specificare il percorso del video
video_path = r"public\test2.mp4"
output_dir = r'input\image'

# aprire il video
cap = cv2.VideoCapture(video_path)

count=0

# iterare attraverso ogni frame del video
while True:
    ret, frame = cap.read()

    # verificare se il video è finito
    if not ret:
        break


    # salvare l'immagine della roi nella cartella "frames"
    output_path = os.path.join(output_dir, 'frame_{}.jpg'.format(int(count)))
    cv2.imwrite(output_path, frame)

    count+=1

    # premere 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# rilasciare la risorsa del video
cap.release()

# chiudere tutte le finestre aperte
cv2.destroyAllWindows()

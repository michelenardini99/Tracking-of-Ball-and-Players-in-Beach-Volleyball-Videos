import cv2
from track_ball import plot_boxes
from roboflow import Roboflow
import numpy as np

rf = Roboflow(api_key="VHujG1OPZBfGofAHKf5a")
project = rf.workspace().project("ball-tracking-beachvolley")
model_ball = project.version(1).model

cap = cv2.VideoCapture(r'public\test2.mp4')

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames in the input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(r'public/output_shot_predictor.mp4',fourcc, fps, (frame_width,frame_height))

posListX = []
posListY = []

xList = [item for item in range(0,1920)]


while True:
    success, img = cap.read()

    if success==True:

        center = plot_boxes(img, model_ball)
        if center:
            center_x, center_y = center
            posListX.append(center_x)
            posListY.append(center_y)
            if len(posListX) > 15:
                posListX.pop(0)
                posListY.pop(0)

        if posListX:
            #Polynomial Regression y = Ax^2 + Bx + C
            #Find the Coefficients
            A, B, C = np.polyfit(posListX,posListY, 2)


            for i, (posx, posy) in enumerate(zip(posListX,posListY)):
                pos=(posx, posy)
                cv2.circle(img, pos, 10, (0,255,0), cv2.FILLED)
                if i==0:
                    cv2.line(img, pos, pos,(0,255,0),5)
                else:
                    cv2.line(img, pos, (posListX[i-1],posListY[i-1]),(0,255,0),5)
        
            for x in xList:
                y = int(A*x**2 + B*x + C)
                cv2.circle(img, (x,y), 2, (255,0,0), cv2.FILLED)

        out_video.write(img)
        cv2.waitKey(50)
    else:
        break

cap.release()
out_video.release()

# Close all windows
cv2.destroyAllWindows()
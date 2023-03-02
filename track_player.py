import cv2
import numpy as np
from utilities import is_inside, closest_position
from player import Player

def detect_player(frame, data):
    net, output_layers, width, height, players, rect2, classes = data
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and (class_id==0):
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = 50
                h = 50
                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            rect1=(x,y,w,h)
            if is_inside(rect1,rect2):
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                temp_player=closest_position(players,np.array([x,y]))
                if(len(players)<4 and temp_player==0):
                    players.append(Player("Player_"+str(len(players)+1),np.array([x,y])))
                    cv2.putText(frame, "Player_"+str(len(players)+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    temp_player.setPos(np.array([x,y]))
                    cv2.putText(frame, temp_player.getName(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame
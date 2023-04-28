import cv2
import numpy as np
from utilities import is_inside, closest_position
from player import Player

def detect_player(frame, data):
    net, output_layers, width, height, players, rect2 = data
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
                """ w = 50
                h = 50
                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8) """
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                rect1=(x,y,w,h)
                if is_inside(rect1,rect2):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            temp_player=closest_position(players,np.array([x,y]))
            if(len(players)<4 and temp_player==0):
                players.append(Player("Player_"+str(len(players)+1),np.array([x,y])))
                #cv2.putText(frame, "Player_"+str(len(players)+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (int(x+w/2), int(y+h)), 3, (255,0,0), 3)
            else:
                temp_player.setPos(np.array([x,y]))
                #cv2.putText(frame, temp_player.getName(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (int(x+w/2), int(y+h)), 3, (255,0,0), 3)

    return frame

def detect_player_pos(frame, data):
    net, output_layers, width, height, players, rect2 = data
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
                """ w = 50
                h = 50
                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8) """
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                rect1=(x,y,w,h)
                if is_inside(rect1,rect2):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    pos=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            pos.append((int(x+w/2),int(y+h)))
            """ confidence = confidences[i]
            temp_player=closest_position(players,np.array([x,y]))
            if(len(players)<4 and temp_player==0):
                players.append(Player("Player_"+str(len(players)+1),np.array([x,y])))
                #cv2.putText(frame, "Player_"+str(len(players)+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (int(x+w/2), int(y+h)), 3, (255,0,0), 3)
            else:
                temp_player.setPos(np.array([x,y]))
                #cv2.putText(frame, temp_player.getName(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (int(x+w/2), int(y+h)), 3, (255,0,0), 3) """
    pos = np.float32(pos)
    return pos
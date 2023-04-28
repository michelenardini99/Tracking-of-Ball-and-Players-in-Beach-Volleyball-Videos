import cv2
import numpy as np


def plot_boxes(frame, model):

    predictions = model.predict(frame, confidence=70, overlap=50)

    for bounding_box in predictions:
        x1 = int(bounding_box['x'] - bounding_box['width'] / 2)
        x2 = int(bounding_box['x'] + bounding_box['width'] / 2)
        y1 = int(bounding_box['y'] - bounding_box['height'] / 2)
        y2 = int(bounding_box['y'] + bounding_box['height'] / 2)
        bgr=(0,255,255)
        cv2.rectangle(frame,(x1,y1), (x2,y2), bgr)
    
    return frame


def get_pos_ball(frame, model): 
    predictions = model.predict(frame, confidence=70, overlap=50)
    pos_ball=[]
    for bounding_box in predictions:
        x1 = int(bounding_box['x'] - bounding_box['width'] / 2)
        x2 = int(bounding_box['x'] + bounding_box['width'] / 2)
        y1 = int(bounding_box['y'] - bounding_box['height'] / 2)
        y2 = int(bounding_box['y'] + bounding_box['height'] / 2)
        if x2>300 and x2<1600:
            center_x = x2-bounding_box['width']/2
            center_y = y2-bounding_box['height']/2
            center = (int(center_x), int(center_y))
            pos_ball.append(center)
            pos_ball = np.float32(pos_ball)
            return pos_ball
    
    return pos_ball 
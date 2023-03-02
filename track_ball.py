import cv2
import torch


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


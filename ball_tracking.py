import pickle
import cv2
import numpy as np
from utilities import is_inside, closest_position, is_above, is_below
from player import Player
from track_ball import plot_boxes
from track_player import detect_player
from roboflow import Roboflow

""" rf = Roboflow(api_key="VHujG1OPZBfGofAHKf5a")
project = rf.workspace().project("volleyball-tracking")
model_ball = project.version(13).model """

rf = Roboflow(api_key="VHujG1OPZBfGofAHKf5a")
project = rf.workspace().project("volley-pose")
model_pose = project.version(1).model

rf = Roboflow(api_key="VHujG1OPZBfGofAHKf5a")
project = rf.workspace().project("ball-tracking-beachvolley")
model_ball = project.version(1).model


# Read the serialized array of vertices from the file
with open("vertex.pickle", "rb") as f:
    vertices_pickled = f.read()

# Deserialize the array of vertices using the pickle module
vertices = pickle.loads(vertices_pickled)

net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
classes = []

# Load coco
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Set up the video capture
cap = cv2.VideoCapture("public/test.mp4")

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames in the input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('public/output_video.mp4',fourcc, fps, (frame_width,frame_height))

players = []
team_1_points=0
team_2_points=0

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_color = (255, 255, 255)
thickness = 2

frame_count=0
time_has_passed=True


if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # Write the frame to the output video file
        height, width, channels = frame.shape
        pts = vertices.reshape((-1,1,2))

        result=str(team_1_points) + "-" + str(team_2_points)

        text_size = cv2.getTextSize(result, font, font_scale, thickness)[0]

        text_x = int((width - text_size[0]) / 2)
        text_y = int(text_size[1] + 10)
        

        # Draw the trapezoid on the copy of the frame
        cv2.polylines(frame,[pts],True,(0,255,0),2)
        rect2=cv2.boundingRect(vertices)

        data = net, output_layers, width, height, players, rect2

        frame = detect_player(frame, data)
        frame = plot_boxes(frame, model_ball)

        predictions = model_pose.predict(frame, confidence=70, overlap=50)

        if time_has_passed==False:
            frame_count+=1
            if frame_count >=50:
                time_has_passed=True


        for bounding_box in predictions:
            x1 = int(bounding_box['x'] - bounding_box['width'] / 2)
            x2 = int(bounding_box['x'] + bounding_box['width'] / 2)
            y1 = int(bounding_box['y'] - bounding_box['height'] / 2)
            y2 = int(bounding_box['y'] + bounding_box['height'] / 2)
            bgr=(0,255,255)
            cv2.rectangle(frame,(x1,y1), (x2,y2), bgr)
            rect1= (x1,y1,bounding_box['width'],bounding_box['height'])
            if bounding_box['class']=='1' and time_has_passed==True:
                if is_above(rect1,rect2):
                    team_1_points+=1
                    time_has_passed=False
                elif is_below(rect1,rect2):
                    team_2_points+=1
                    time_has_passed=False

        cv2.putText(frame, result, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        out_video.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break




# Release the video capture and writer objects
cap.release()
out_video.release()

# Close all windows
cv2.destroyAllWindows()

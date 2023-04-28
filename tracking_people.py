import cv2
import argparse
import sys
import numpy as np
import os.path
import math
import torch
import json
import pickle
from utilities import is_inside
import torchvision
from torchvision import transforms as T
from centroidtracker import CentroidTracker
from object_trackable import TrackableObject
# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
        
# Load names of classes
coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
"frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
"baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
"plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
"banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
"pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
"laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
"oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
"clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]
font = cv2.FONT_HERSHEY_SIMPLEX

with open("vertex.pickle", "rb") as f:
    vertices_pickled = f.read()

# Deserialize the array of vertices using the pickle module
vertices = pickle.loads(vertices_pickled)
pts = vertices.reshape((-1,1,2))
print(pts)
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
model.eval()

cap = cv2.VideoCapture("public/test.mp4")

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames in the input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('public/output_video_player_court_detected.mp4',fourcc, fps, (frame_width,frame_height))


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame_data, pred):
    frame, count = frame_data
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    num = torch.argwhere(scores > confThreshold).shape[0]
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i]  - 1]
        if(class_name == "person"):
            classId = labels.numpy()[i] - 1
            confidence = scores.numpy()[i]
            center_x = int((x1 + x2)/2)
            center_y = int((y1 + y2)/2)
            width = int(x2 - x1)
            height = int(y2 - y1)
            left = int(x1)
            top = int(y1)
            player=(center_x, center_y)
            if is_inside(player,court):
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            counting(objects, count)


def counting(objects,frame_count):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
 
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            
 
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalUp += 1
                    to.counted = True
 
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalDown += 1
                    to.counted = True
 
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        center_x = int((centroid[0] + centroid[2])/2)
        center_y = int((centroid[3]))
        color = ( int (to.color [ 0 ]), int (to.color [ 1 ]), int (to.color [ 2 ]))
        """ cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) """
        cv2.circle(frame, (int((centroid[0] + centroid[2])/2), int((centroid[1] + centroid[3])/2)), 10, color, -1)
        player_json = {
            "frame": count,
            "id": objectID,
            "x": center_x,
            "y": center_y,
            "color": color
        }
        write_json(player_json)
    # construct a tuple of information we will be displaying on the
    # frame

def write_json(new_data, filename='player_pos.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["player"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)


count = 0
court=cv2.boundingRect(vertices)
while cap.isOpened():
    ret, frame = cap.read()
    frame_c = frame
    if ret == True and count<50:
        count+=1
        mask = np.zeros_like(frame_c) 
        transform = T.ToTensor()
        img = transform(frame_c)
        
        with torch.no_grad():
            """ start = timeit.default_timer() """
            pred = model([img])
            """ end = timeit.default_timer() """
            
            # Remove the bounding boxes with low confidence
            frame_data = frame, count
            postprocess(frame_data, pred)
            cv2.polylines(frame,[pts],True,(0,255,0),2)
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



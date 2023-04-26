from sort import *
from tracker import Tracker
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import cv2
import timeit


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

""" mot_tracker = Sort()  """

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
out_video = cv2.VideoWriter('public/output_video_SORT.mp4',fourcc, fps, (frame_width,frame_height))
count=0
if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    frame_c = frame
    if ret == True and count<500:
        count+=1
        transform = T.ToTensor()
        img = transform(frame_c)

        with torch.no_grad():
            """ start = timeit.default_timer() """
            pred = model([img])
            """ end = timeit.default_timer() """
            bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
            track_bbs_ids = mot_tracker.update(bboxes)
            num = torch.argwhere(scores > 0.5).shape[0]
            for i in range(num):
                x1, y1, x2, y2, id = track_bbs_ids[i].astype("int")
                class_name = coco_names[labels.numpy()[i]  - 1]
                if(class_name == "person"):
                    """ cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA) """
                    cv2.putText(frame, str(id), (x1, y1), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

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

""" print("Tempo di elaborazione:", round(end-start,3), "secondi") """
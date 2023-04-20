import cv2
import numpy as np
import matplotlib.pyplot as plt
import track_player as tp

HALF_COURT = 305

im_src = cv2.imread(r'public\ball.png')

net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
classes = []

# Load coco
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("public/test.mp4")

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames in the input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



TOP_LEFT = (663, 223)
TOP_RIGHT = (1525, 242)
BOT_LEFT = (560, 1089)
BOT_RIGHT = (1833, 1089)


vertices = [TOP_LEFT, TOP_RIGHT, BOT_RIGHT, BOT_LEFT]
pts_src = np.float32(vertices)

players = []

height, width, channels = im_src.shape
rect2=cv2.boundingRect(pts_src)

data = net, output_layers, width, height, players, rect2




# Read destination image.
im_dst = cv2.imread(r'public\court.jpg')

height_dest, width_dest, channels = im_dst.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('public/minimap.mp4',fourcc, fps, (width_dest,height_dest))
# Four corners of the book in destination image.
pts_dst = np.array([[89, 48],[348, 48],[348, 564],[89, 564]])

pos_text=[]


while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:


        im_dst = cv2.imread(r'public\court.jpg')
        pos = tp.detect_player_pos(frame, data)

        

        if len(pos) != 0:
            # Calculate Homography
            H, _ = cv2.findHomography(pts_src, pts_dst)

            # Warp source image to destination based on homography
            transformed_positions = cv2.perspectiveTransform(pos.reshape(len(pos), 1, 2), H)


            for position in transformed_positions:
                x, y = position[0]
                pos_text.append(str(x) + ' ' + str(y) + '\n')
                if y > HALF_COURT: color = (255, 0, 0)
                else: color = (0, 0, 255)
                cv2.circle(im_dst, (int(x), int(y)), 10, color, -1)

        out_video.write(im_dst)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        with open(r'output\player_pos_heatmap.txt', 'w') as f:
            f.writelines(pos_text)
        break
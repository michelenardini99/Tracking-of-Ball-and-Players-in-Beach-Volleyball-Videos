import cv2
import numpy as np
import matplotlib.pyplot as plt
import track_player as tp
import json

HALF_COURT = 305

im_src = cv2.imread(r'public\ball.png')

cap = cv2.VideoCapture("public/test.mp4")
# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames in the input video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



TOP_LEFT = (529, 186)
TOP_RIGHT = (1271, 201)
BOT_LEFT = (464, 908)
BOT_RIGHT = (1529, 908)


vertices = [BOT_LEFT, TOP_LEFT, TOP_RIGHT, BOT_RIGHT]
pts_src = np.float32(vertices)

players = []

height, width, channels = im_src.shape
rect2=cv2.boundingRect(pts_src)





# Read destination image.
im_dst = cv2.imread(r'public\court.jpg')

height_dest, width_dest, channels = im_dst.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('public/minimap.mp4',fourcc, fps, (width_dest,height_dest))
# Four corners of the book in destination image.
pts_dst = np.array([[90, 564],[90, 48],[348, 48],[348, 564]])

f = open('player_pos.json')

player_pos = json.load(f)

count = 0
while True:
    count += 1
    if count < 50:


        im_dst = cv2.imread(r'public\court.jpg')
        
        pos = []
        color = []
        for player in player_pos["player"]:
            if player["frame"] == count:
               pl_pos = (player["x"], player["y"])
               pos.append(pl_pos)
               color.append(player["color"])
        pos=np.float32(pos)
        if len(pos) != 0:
            # Calculate Homography
            H, _ = cv2.findHomography(pts_src, pts_dst)

            # Warp source image to destination based on homography
            transformed_positions = cv2.perspectiveTransform(pos.reshape(len(pos), 1, 2), H)


            for index, position in enumerate(transformed_positions):
                x, y = position[0]
                pl_color = color[index]
                cv2.circle(im_dst, (int(x), int(y)), 10, pl_color, -1)

        out_video.write(im_dst)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
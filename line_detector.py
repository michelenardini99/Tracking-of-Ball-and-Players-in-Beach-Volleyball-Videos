import cv2
import numpy as np
from utilities import get_top_left_angle, get_top_right_angle, get_bottom_left_angle, get_bottom_right_angle

video = cv2.VideoCapture(r"public\test.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture(r"public\test.mp4")
        continue
    h, w, _ = orig_frame.shape

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    mask = cv2.inRange(hsv, low_blue, high_blue)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    top_left = get_top_left_angle(lines, w)
    top_right = get_top_right_angle(lines, w)
    bottom_right = get_bottom_right_angle(lines, w)
    bottom_left = get_bottom_left_angle(lines, w)

    trapezoid_points = [top_left, top_right, bottom_right, bottom_left]
    trapezoid_points = np.array(trapezoid_points, np.int32)

    cv2.polylines(frame, [trapezoid_points], True, (0, 255, 0), thickness=3)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
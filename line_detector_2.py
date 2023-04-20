import cv2
import numpy as np

img = cv2.imread(r"public\partita-di-calcio_2.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255)) # green mask to select only the field
frame_masked = cv2.bitwise_and(img, img, mask=mask_green)
gray = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

canny = cv2.Canny(gray, 50, 150, apertureSize=3)
# Hough line detection
lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 20)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("Edges", canny)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
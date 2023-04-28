import cv2

# Load the image
img = cv2.imread(r'public\ball.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours
max_area = 0
max_rect = None
for cnt in contours:
    # Filter out small contours
    if cv2.contourArea(cnt) < 100:
        continue

    # Approximate the contour as a polygon
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

    # Check if the polygon is a rectangle
    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue

    # Compute the area of the rectangle
    area = cv2.contourArea(approx)

    # Keep track of the largest rectangle found
    if area > max_area:
        max_area = area
        max_rect = approx

# Draw the largest rectangle on the image
cv2.drawContours(img, [max_rect], 0, (0, 255, 0), 2)

# Show the image
cv2.imshow('Largest Rectangle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

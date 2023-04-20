import cv2
import numpy as np
import pickle

# Set up the video capture
cap = cv2.imread(r"public\court.jpg")

""" # Read the first frame
ret, frame = cap.read() """

# Create a copy of the frame
frame_copy = cap.copy()

# Define the four vertices of the trapezoid
trap_vertices = []

def on_mouse_click(event, x, y, flags, param):
    global trap_vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        trap_vertices.append((x, y))
        print(x)
        print(y)
        if len(trap_vertices) == 4:
            draw_trapezoid()

def draw_trapezoid():
    global frame_copy
    global trap_vertices

    # Create an array of the trapezoid's vertices
    pts = np.array([trap_vertices], np.int32)
    vertices_pickled = pickle.dumps(pts)

# Write the serialized array of vertices to a file
    with open("vertex.pickle", "wb") as f:
        f.write(vertices_pickled)
    # reshape the array in the form required by polylines
    pts = pts.reshape((-1,1,2))

    # Draw the trapezoid on the copy of the frame
    cv2.polylines(frame_copy,[pts],True,(0,255,0),2)

    # Show the frame with the trapezoid
    cv2.imshow("Frame", frame_copy)

# Create a window to show the frame
cv2.namedWindow("Frame")

# Set the mouse callback function
cv2.setMouseCallback("Frame", on_mouse_click)

# Show the frame
cv2.imshow("Frame", cap)

# Wait for the user to press a key
cv2.waitKey(0)

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

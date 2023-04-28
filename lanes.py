import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 255, 0], thickness=2):
    """    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)               
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def display_img(img):
    """
    This function is used to display images inline which are read using cv2.
    
    cv2 reads image in BGR where as matplotlib in RGB
    
    using cv2.cvtColor() we can use matplotlib.plot.imshow()
    
    'img' is image read by cv2.imread() in BGR format
    """
    if img.ndim > 2:
        plt.imshow(img)
        plt.show()
    else:
        plt.imshow(img, cmap = 'gray')
        plt.show()

def process_image_challenge(image):
    
    # Get the original Image
    input_image = image
                                 
    # Check if the image was loaded properly
    if input_image is None: 
        raise Exception("could not load image !")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    # Canny edge detection
    canny_image = canny(mask, 150, 240)

    display_img(canny_image)
    
    # ROI
    vertices = np.array([[(433, 933), (491, 162), (1292, 162), (1560, 933)]])
    ROI_image = region_of_interest(canny_image, vertices)

    
    
    # Hough lines
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 20
    max_line_gap = 20
    
    hough_image, lines = hough_lines(ROI_image, rho, theta, threshold, min_line_length, max_line_gap)

    

    
    result = weighted_img(hough_image, input_image)

    
    return result

image = cv2.imread("public/frame_1.jpg")

res = display_img(process_image_challenge(image))



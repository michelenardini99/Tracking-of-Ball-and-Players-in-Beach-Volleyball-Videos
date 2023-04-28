from PIL import Image, ImageDraw
import cv2

width = 240
height = 480

img  = Image.new( mode = "RGB", size = (width, height), color = (255,255,255) )
draw = ImageDraw.Draw(img)
draw.line((0,int(height/2),(int(width), int(height/2))), fill=128)
img.save("public/court.jpeg")
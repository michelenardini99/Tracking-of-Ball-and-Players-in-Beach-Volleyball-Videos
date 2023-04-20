from player import Player
import numpy as np

TOL=100

def is_inside(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 > x2 and y1 > y2-TOL and x1 + w1 < x2 + w2

def is_above(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return y1 < y2

def is_below(rect1,rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return y1 > y2


def closest_position(players, input_position):
    if(len(players)==0):
        return 0
    pos=[]
    for player in players:
        pos.append(player.getCoord())

    distances = np.linalg.norm(pos - input_position, axis=1)
    closest_index = np.argmin(distances)
    if (distances[closest_index]<20 and len(players)<4) or (len(players)==4):
        return players[closest_index]
    else:
        return 0

def get_top_right_angle(lines, w):
    highest_line = None
    highest_point = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (highest_line is None or y1 < highest_line[0][1]):
            highest_line = line
            highest_point = (x1, y1)
        if y2 < highest_line[0][1] and x2 > w/2:
            highest_line = line
            highest_point = (x2, y2)
    return highest_point

def get_top_left_angle(lines, w):
    highest_line = None
    highest_point = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (highest_line is None or y1 < highest_line[0][1]):
            highest_line = line
            highest_point = (x1, y1)
        if y2 < highest_line[0][1] and x2 < w/2:
            highest_line = line
            highest_point = (x2, y2)
    return highest_point

def get_bottom_right_angle(lines, w):
    lowest_line = None
    lowest_point = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 > w/2 and x2 > w/2:  # Controlla che la linea sia nella metà destra della ROI
            if lowest_line is None or y1 > lowest_line[0][1]:
                lowest_line = line
                lowest_point = (x1, y1)
            if y2 > lowest_line[0][1]:
                lowest_line = line
                lowest_point = (x2, y2)
    return lowest_point

def get_bottom_left_angle(lines, w):
    lowest_line = None
    lowest_point = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < w/2 and x2 < w/2:  # Controlla che la linea sia nella metà destra della ROI
            if lowest_line is None or y1 > lowest_line[0][1]:
                lowest_line = line
                lowest_point = (x1, y1)
            if y2 > lowest_line[0][1]:
                lowest_line = line
                lowest_point = (x2, y2)
    return lowest_point


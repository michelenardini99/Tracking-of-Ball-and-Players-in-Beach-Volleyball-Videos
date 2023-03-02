from player import Player


import numpy as np

def is_inside(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2

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

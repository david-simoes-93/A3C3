import numpy as np


# get distance between points A and B
def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# calculate new speed value when ball hits something
def bounce_speed(speed):
    speed *= 100
    bounce_spd = 0.0001 * speed ** 2 - 0.3785 * speed - 0.5477
    if abs(bounce_spd) < 1.3:
        bounce_spd = 0
    else:
        bounce_spd /= 100
    return bounce_spd


# returns whether a rectangle with center at rectangle_pos and rectangle_size=[width,height]
# intersects a circle at reward_pos with radius reward_radius
def intersects(rectangle_pos, rectangle_size, reward_pos, reward_radius):
    circle_distance = [abs(reward_pos[0] - rectangle_pos[0]), abs(reward_pos[1] - rectangle_pos[1])]

    if circle_distance[0] > rectangle_size[0] / 2 + reward_radius:
        return False
    if circle_distance[1] > rectangle_size[1] / 2 + reward_radius:
        return False

    if circle_distance[0] <= rectangle_size[0] / 2:
        return True
    if circle_distance[1] <= rectangle_size[1] / 2:
        return True

    corner_distance_sq = (circle_distance[0] - rectangle_size[0] / 2) ** 2 + \
                         (circle_distance[1] - rectangle_size[1] / 2) ** 2

    return corner_distance_sq <= reward_radius ** 2

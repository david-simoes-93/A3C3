import pygame
from simulator_geof2.Players.Agent import Agent
from simulator_geof2.Simulator.Utils import intersects
from gym.spaces import *

from simulator_geof2.MapGenerators import Obstacle


class Rectangle(Agent):
    def __init__(self, can_interrupt_growth=True):
        #self.frameskip = frameskip
        self.can_interrupt_growth = can_interrupt_growth
        self.action_space = Discrete(4)

        self.pos = None
        self.growing_side = False  # upwards
        self.rect_min, self.rect_max = 40, 200
        self.rect_w, self.rect_h = self.rect_max, self.rect_min
        self.rectangle_ground = None

        # if movements are repeated, what should they become (like, you won't resize N times, resize once then NoOp)
        self.repeated_movement_indexes = [0, 1, 3, 3]

    def step(self, action_rectangle):
        can_grow_side = self.rect_w - 2 < self.rect_max
        can_grow_up = self.rect_w - 2 > self.rect_min

        # Rectangle movement
        if action_rectangle == 0:  # LEFT
            self.pos[0] -= 5
        elif action_rectangle == 1:  # RIGHT
            self.pos[0] += 5
        elif action_rectangle == 2:  # RESIZE
            if self.can_interrupt_growth:
                self.growing_side = not self.growing_side
            else:
                if not can_grow_up:
                    self.growing_side = True
                elif not can_grow_side:
                    self.growing_side = False
        elif action_rectangle == 3:  # NOTHING
            pass

        self.pos[1] += 3

        if not self.growing_side and can_grow_up:
            # if can grow upwards
            self.pos[1] -= 1
            self.rect_w = self.rect_w - 2
            self.rect_h = self.rect_h + 2
        elif self.growing_side and can_grow_side:
            # if can grow sideways
            self.pos[1] += 1
            self.rect_w = self.rect_w + 2
            self.rect_h = self.rect_h - 2

    def clear_out_of_obstacles(self, obstacles, forbidden_moves=[False, False, False, False]):
        already_moved = [False, False, False, False]

        # move rectangle out of obstacles
        for obs in obstacles:
            # obstacle smaller than rectangle
            if self.pos[0] - self.rect_w / 2 < obs.center_x < self.pos[0] + self.rect_w / 2:
                if not forbidden_moves[2] and obs.top_y < self.pos[1] + self.rect_h / 2 < obs.top_y + 3.01:  # ground
                    self.pos[1] = obs.top_y - self.rect_h / 2
                    already_moved[3] = True

            # left part of rectangle inside obstacle
            if obs.left_x < self.pos[0] - self.rect_w / 2 < obs.right_x:
                if not forbidden_moves[2] and obs.top_y < self.pos[1] + self.rect_h / 2 < obs.top_y + 3.01:  # ground
                    self.pos[1] = obs.top_y - self.rect_h / 2
                    already_moved[3] = True

            # right part of rectangle inside obstacle
            if obs.left_x < self.pos[0] + self.rect_w / 2 < obs.right_x:
                if not forbidden_moves[2] and obs.top_y < self.pos[1] + self.rect_h / 2 < obs.top_y + 3.01:  # ground
                    self.pos[1] = obs.top_y - self.rect_h / 2
                    already_moved[3] = True

            # rectangle height catches obstacle
            if self.pos[1] - self.rect_h / 2 < obs.center_y < self.pos[1] + self.rect_h / 2:
                if not forbidden_moves[1] and obs.left_x < self.pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                    self.pos[0] = obs.right_x + self.rect_w / 2
                    already_moved[0] = True
                elif not forbidden_moves[0] and obs.left_x < self.pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                    self.pos[0] = obs.left_x - self.rect_w / 2
                    already_moved[1] = True

            # bottom part of rectangle inside obstacle
            if obs.top_y < self.pos[1] + self.rect_h / 2 < obs.bot_y:
                if not forbidden_moves[1] and obs.left_x < self.pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                    self.pos[0] = obs.right_x + self.rect_w / 2
                    already_moved[0] = True
                elif not forbidden_moves[0] and obs.left_x < self.pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                    self.pos[0] = obs.left_x - self.rect_w / 2
                    already_moved[1] = True

            # top part of rectangle inside obstacle
            if obs.top_y < self.pos[1] - self.rect_h / 2 < obs.bot_y:
                if not forbidden_moves[1] and obs.left_x < self.pos[0] - self.rect_w / 2 < obs.right_x:  # left wall
                    self.pos[0] = obs.right_x + self.rect_w / 2
                    already_moved[0] = True
                elif not forbidden_moves[0] and obs.left_x < self.pos[0] + self.rect_w / 2 < obs.right_x:  # right wall
                    self.pos[0] = obs.left_x - self.rect_w / 2
                    already_moved[1] = True

        self.set_on_ground_rectangle(obstacles)

        return already_moved

    def check_rewards(self, rewards):
        # intersected rewards
        intersected_rewards = []
        for reward in rewards:
            if intersects(self.pos, [self.rect_w, self.rect_h], reward, 25):
                intersected_rewards.append(reward)

        return intersected_rewards

    def reset(self, obstacles):
        self.rect_w, self.rect_h = self.rect_max, self.rect_min
        self.growing_side = True  # growing sideways
        self.set_on_ground_rectangle(obstacles)

    def render(self, screen):
        pygame.draw.rect(screen, (0, 255, 0),
                         [int(self.pos[0] - self.rect_w / 2),
                          int(self.pos[1] - self.rect_h / 2),
                          self.rect_w, self.rect_h])

    # checks and sets if rectangle is touching the ground
    def set_on_ground_rectangle(self, obstacles):
        index_g = -1  # default ground
        highest_ground = 800

        for i, obs in enumerate(obstacles):
            if (obs.left_x < self.pos[0] - self.rect_w / 2 < obs.right_x or
                            obs.left_x < self.pos[0] + self.rect_w / 2 < obs.right_x):
                if self.pos[1] < obs.top_y < highest_ground:
                    index_g = i
                    highest_ground = obstacles[index_g].top_y
                elif self.pos[1] < obs.top_y == highest_ground:
                    # multiple grounds, pick the one with most area
                    if obs.center_x < obstacles[index_g].center_x and \
                                    abs(obs.right_x - self.pos[0]) < abs(
                                        obstacles[index_g].right_x - self.pos[0]):
                        index_g = i

        if index_g!=-1:
            self.rectangle_ground = obstacles[index_g]

    # Returns the Rectangle's state: [position (X), position (Y), rectangle growing sideways (bool), size (X), size (Y)]
    def get_state(self):
        state = [self.pos[0], self.pos[1], self.growing_side, self.rect_w, self.rect_h]
        return state

    # Returns the Rectangle's position and shape for other agents: [position (X), position (Y), size (X), size (Y)]
    def get_external_state(self):
        state = [self.pos[0], self.pos[1], self.rect_w, self.rect_h]
        return state

    def get_obstacle_body(self):
        return [Obstacle(self.pos, self.rect_w, self.rect_h)]
import numpy as np
import pygame
from simulator_geof2.Players.Agent import Agent
from simulator_geof2.Simulator.Utils import bounce_speed, intersects, distance
from gym.spaces import *

from simulator_geof2.MapGenerators import Obstacle


class Circle(Agent):
    def __init__(self, air_movement=False):
        self.pos = None
        self.action_space = Discrete(4)

        self.circle_vel = []
        self.circle_spin = 0
        self.circle_on_ground = False
        self.circle_ground = None
        self.circle_radius = 40

        # self.frameskip = frameskip
        self.air_movement = air_movement

        # if movements are repeated, what should they become (like, you won't jump N times, jump once then NoOp)
        self.repeated_movement_indexes = [3, 1, 2, 3]

    def step(self, action_circle):
        self.circle_spin *= 0.99

        # Circle movement
        if action_circle == 0:  # LEFT
            self.circle_spin = self.circle_spin - 0.02
        elif action_circle == 1:  # RIGHT
            self.circle_spin = self.circle_spin + 0.02
        elif action_circle == 2 and self.circle_on_ground:  # JUMP
            self.circle_vel[1] = -4.40
        elif action_circle == 3:  # NOTHING
            pass

        # move on air if allowed, or only move while on ground
        if self.air_movement or self.circle_on_ground:
            self.circle_vel[0] = self.circle_spin
        # gravity
        self.circle_vel[1] += 0.03

        self.pos[0] += self.circle_vel[0]
        self.pos[1] += self.circle_vel[1]

    def clear_out_of_obstacles(self, obstacles, forbidden_moves=[False, False, False, False]):
        already_moved = [False, False, False, False]

        # move circle out of obstacles
        current_moved_x, current_moved_y = 0, 0
        for obs in obstacles:
            if intersects([obs.center_x, obs.center_y], [obs.half_width * 2, obs.half_height * 2],
                          self.pos, self.circle_radius - 0.01):
                # circle crossed horizontal lines of obstacle
                if obs.left_x < self.pos[0] < obs.right_x:
                    # fell inside obstacle below
                    if obs.top_y < self.pos[1] + self.circle_radius < obs.bot_y:
                        if not forbidden_moves[2]:
                            current_moved_y += (obs.top_y - self.circle_radius) - self.pos[1]
                            self.pos[1] = obs.top_y - self.circle_radius
                            self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                            already_moved[3] = True
                    # jumped into obstacle above
                    else:
                        if not forbidden_moves[3]:
                            current_moved_y += (obs.bot_y + self.circle_radius) - self.pos[1]
                            self.pos[1] = obs.bot_y + self.circle_radius
                            self.circle_vel[1] = bounce_speed(self.circle_vel[1])
                            already_moved[2] = True
                # circle crossed vertical lines of obstacle
                elif obs.top_y < self.pos[1] < obs.bot_y:
                    # inside a wall on circle's left
                    if obs.left_x < self.pos[0] - self.circle_radius < obs.right_x:
                        if not forbidden_moves[1]:
                            current_moved_x += (obs.right_x + self.circle_radius) - self.pos[0]
                            self.pos[0] = obs.right_x + self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                            already_moved[0] = True
                    # inside a wall on circle's right
                    else:
                        if not forbidden_moves[0]:
                            current_moved_x += (obs.left_x - self.circle_radius) - self.pos[0]
                            self.pos[0] = obs.left_x - self.circle_radius
                            self.circle_vel[0] = bounce_speed(self.circle_vel[0])
                            self.circle_spin = bounce_speed(self.circle_spin)
                            already_moved[1] = True
                # some corner interception
                else:
                    # the mod_x and mod_y allow us to treat all 4 corners in the same way and then just invert x or y as necessary
                    mod_x = 1 if obs.center_x - self.pos[0] > 0 else -1
                    mod_y = 1 if obs.center_y - self.pos[1] > 0 else -1
                    dist_x = np.abs(obs.center_x - self.pos[0]) - obs.half_width
                    dist_y = np.abs(obs.center_y - self.pos[1]) - obs.half_height

                    # if distance in Y is larger (so circle is more up or down), we change Y (we affect height)
                    if dist_x < dist_y:
                        new_dist_x = 0
                        new_dist_y = np.sqrt(self.circle_radius ** 2 - dist_x ** 2) - dist_y
                    # else if distance in X is larger (so circle is more to the side), we change X (we affect width)
                    else:
                        new_dist_x = np.sqrt(self.circle_radius ** 2 - dist_y ** 2) - dist_x
                        new_dist_y = 0

                    # if movement to dodge corner contradicts movement to avoid another obstacle, we change x and y
                    if new_dist_x * current_moved_x < 0:
                        new_dist_y = new_dist_x
                    if new_dist_y * current_moved_y < 0:
                        new_dist_x = new_dist_y

                    current_moved_x += (new_dist_x * -mod_x)
                    if not forbidden_moves[1] and new_dist_x * -mod_x > 0:
                        self.pos[0] += new_dist_x * -mod_x
                        already_moved[0] = True
                    elif not forbidden_moves[0] and new_dist_x * -mod_x < 0:
                        self.pos[0] += new_dist_x * -mod_x
                        already_moved[1] = True
                    current_moved_y += (new_dist_y * -mod_y)
                    if not forbidden_moves[2] and new_dist_y * -mod_y > 0:
                        self.pos[1] += new_dist_y * -mod_y
                        already_moved[3] = True
                    elif not forbidden_moves[3] and new_dist_y * -mod_y < 0:
                        self.pos[1] += new_dist_y * -mod_y
                        already_moved[2] = True

        self.set_on_ground_circle(obstacles)

        return already_moved

    def check_rewards(self, rewards):
        # intersected rewards
        intersected_rewards = []
        for i in range(len(rewards)):
            if distance(rewards[i], self.pos) < self.circle_radius + 25:
                intersected_rewards.append(rewards[i])

        return intersected_rewards

    # Returns the Circle's state: [position (X), position (Y), velocity (X), velocity (Y)]
    def get_state(self):
        state = [self.pos[0], self.pos[1], self.circle_vel[0], self.circle_vel[1]]
        return state

    # Returns the Circle's position for other agents: [position (X), position (Y)]
    def get_external_state(self):
        state = [self.pos[0], self.pos[1]]
        return state

    def reset(self, obstacles):
        self.circle_vel = [0, 0]
        self.circle_spin = 0
        self.set_on_ground_circle(obstacles)

    def render(self, screen):
        pygame.draw.circle(screen, (255, 255, 0),
                           [int(self.pos[0]), int(self.pos[1])],
                           self.circle_radius)

    # checks and sets if circle is touching the ground
    def set_on_ground_circle(self, obstacles):
        found_ground = self.get_ground_circle(obstacles)
        if found_ground is None:
            return

        self.circle_ground = found_ground

        # if completely on top and y=40, on ground
        if self.circle_ground.left_x < self.pos[0] < self.circle_ground.right_x:
            self.circle_on_ground = self.circle_ground.top_y - self.pos[1] < self.circle_radius + 0.01
        # if to the side and d=40, on ground
        elif self.pos[0] < self.circle_ground.left_x:
            self.circle_on_ground = distance([self.circle_ground.left_x, self.circle_ground.top_y],
                                             self.pos) < self.circle_radius + 0.01
        # if to the side and d=40, on ground
        elif self.circle_ground.right_x < self.pos[0]:
            self.circle_on_ground = distance([self.circle_ground.right_x, self.circle_ground.top_y],
                                             self.pos) < self.circle_radius + 0.01

    # returns the closest platform below the circle
    def get_ground_circle(self, obstacles):
        index = -1  # default ground
        highest_ground_position = 800

        for i, obs in enumerate(obstacles):
            if obs.left_x - self.circle_radius < self.pos[0] < obs.right_x + self.circle_radius and \
                                    self.pos[1] < obs.top_y < highest_ground_position:
                index = i
                highest_ground_position = obstacles[index].top_y

        if index != -1:
            return obstacles[index]
        return None

    def get_obstacle_body(self):
        return [Obstacle(self.pos, 12, self.circle_radius * 2), Obstacle(self.pos, self.circle_radius * 2, 12),
                Obstacle(self.pos, 33, self.circle_radius * 2 - 8), Obstacle(self.pos, self.circle_radius * 2 - 8, 33),
                Obstacle(self.pos, 56, 56)]

#!/usr/bin/python3
from time import sleep

import gym
import pygame
from gym.spaces import *
import random
import sys
import numpy as np


class GymTraffic(gym.Env):
    def __init__(self, intersections=6, number_of_agents=10, road_size=3, frequency=0.5):
        # super?
        self.max_actions = 2
        self.number_of_agents = number_of_agents
        self.road_size = road_size
        self.intersections = intersections
        self.frequency = frequency

        self.grid_size = road_size * intersections + road_size - 1
        self.screen = None
        self.agent_colors = []

        # Global info
        self.rewards = []
        self.terminal = False

        self.pos = {}
        self.turn = [0] * number_of_agents
        self.prev_rewards = [0] * number_of_agents
        self.already_moved = [False] * number_of_agents
        self.intersection_crash = [False] * number_of_agents
        self.timer = 0

        self.intersection_collision_counter = 0
        self.stall_counter = 0

        # GUI
        self.metadata = {'render.modes': ['human']}

        # Public GYM variables
        # The Space object corresponding to valid actions
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(self.max_actions) for _ in range(self.number_of_agents)])
        # (gym.spaces.Discrete(self.max_actions), gym.spaces.Discrete(self.max_actions)))
        # The Space object corresponding to valid observations
        # turning, before, after, left
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(number_of_agents, 4))

        self.agent_observation_space = [4]
        self.central_observation_space = [7]
        self.agent_action_space = self.max_actions

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, 1]
        self.map_border = self.road_size * self.intersections + self.road_size - 1

    def render(self, mode='human', close=False):
        cell_width = 50
        cell_width_half = int(cell_width / 2)
        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GymTraffic, self).render(mode=mode)
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode([cell_width * self.grid_size, cell_width * self.grid_size])
            pygame.display.set_caption("Traffic")
            for _ in self.pos:
                self.agent_colors.append(
                    (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)))

        self.screen.fill((255, 255, 255))

        # Draw roads
        for x in range(self.intersections):
            for y in range(self.grid_size):
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 [(x * self.road_size + self.road_size - 1) * cell_width, y * cell_width, cell_width,
                                  cell_width])
        for y in range(self.intersections):
            for x in range(self.grid_size):
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 [x * cell_width, (y * self.road_size + self.road_size - 1) * cell_width, cell_width,
                                  cell_width])

        # Draw agents
        for agent_index, agent in enumerate(self.pos.values()):
            pygame.draw.circle(self.screen, self.agent_colors[agent_index],
                               [agent[0] * cell_width + cell_width_half, agent[1] * cell_width + cell_width_half],
                               cell_width_half)
            if self.turn[agent_index] == 1:
                pygame.draw.circle(self.screen, (0, 0, 0),
                                   [agent[0] * cell_width + cell_width_half, agent[1] * cell_width + cell_width_half],
                                   2)
            if self.intersection_crash[agent_index]:
                pygame.draw.circle(self.screen, (255, 0, 0),
                                   [agent[0] * cell_width + cell_width_half, agent[1] * cell_width + cell_width_half],
                                   4)

        pygame.display.flip()
        # sleep(2)

    def reset(self):
        self.intersection_collision_counter = 0
        self.stall_counter = 0

        prev_lane = 0
        prev_pos = 0
        for i in range(self.number_of_agents):
            lane = int(i / (self.number_of_agents / (self.intersections * 2)))
            if lane != prev_lane:
                prev_pos = 0
            prev_lane = lane

            if lane < self.intersections:  # horizontal lane
                self.pos[i] = [prev_pos, lane * self.road_size - 1 + self.road_size]
            else:  # vertical lane
                self.pos[i] = [(lane - self.intersections) * self.road_size - 1 + self.road_size, prev_pos]
            self.turn[i] = random.randint(0, 1)
            # print(self.pos[i], self.turn[i])

            prev_pos -= 1
            while random.random() > self.frequency:
                prev_pos -= 1

        self.prev_rewards = [0] * self.number_of_agents
        self.already_moved = [False] * self.number_of_agents
        self.intersection_crash = [False] * self.number_of_agents
        states = self.get_state()
        return states, {"state_central": self.get_central_state(states)}

    def step(self, actions):
        self.already_moved = [False] * self.number_of_agents
        self.intersection_crash = [False] * self.number_of_agents
        rewards = [0.1] * self.number_of_agents
        terminal = True

        # returns agent indexes by order of who's in front
        for i in sorted(self.pos, key=lambda val: -self.pos[val][0] - self.pos[val][1]):
            action = actions[i]
            pos = self.pos[i]

            if pos[0] < self.map_border or pos[1] < self.map_border:
                terminal = False

            if pos[0] >= self.map_border or pos[1] >= self.map_border:
                pos[0] = self.map_border
                pos[1] = self.map_border
                continue

            # before map, move up or right as appropriate
            if pos[0] <= 0:
                if [pos[0] + 1, pos[1]] not in self.pos.values():
                    pos[0] += 1
                continue
            elif pos[1] <= 0:
                if [pos[0], pos[1] + 1] not in self.pos.values():
                    pos[1] += 1
                continue

            # action: stop
            # penalize for stopping, count if stopped at intersection
            if action == 0:
                rewards[i] = -0.5
                if pos[1] % self.road_size == self.road_size - 2:
                    # at crossroads, vertical lane
                    self.stall_counter += 1
                    # rewards[i] = -2
                elif pos[0] % self.road_size == self.road_size - 2:
                    # at crossroads, horizontal lane
                    self.stall_counter += 1
                    # rewards[i] = -2
                continue

            self.already_moved[i] = True
            # move forward
            # heavy penalty if crashed into some other car which went the same way
            # penalize harder if you did not have priority (people turning have priority)
            if pos[1] % self.road_size == self.road_size - 2:
                # at crossroads, vertical lane
                if self.turn[i] == 0:
                    if [pos[0], pos[1] + 2] in self.pos.values():
                        rewards[i] = -10
                        if [pos[0], pos[1] + 2] in self.already_moved:
                            self.intersection_collision_counter += 1
                            self.intersection_crash[i] = True
                    else:
                        pos[1] += 2
                        self.turn[i] = random.randint(0, 1)
                        self.already_moved[i] = pos
                else:
                    if [pos[0] + 1, pos[1] + 1] in self.pos.values():
                        rewards[i] = -5
                        if [pos[0] + 1, pos[1] + 1] in self.already_moved:
                            self.intersection_collision_counter += 1
                            self.intersection_crash[i] = True
                    else:
                        pos[0] += 1
                        pos[1] += 1
                        self.turn[i] = random.randint(0, 1)
                        self.already_moved[i] = pos
            elif pos[0] % self.road_size == self.road_size - 2:
                # at crossroads, horizontal lane
                if self.turn[i] == 0:
                    if [pos[0] + 2, pos[1]] in self.pos.values():
                        rewards[i] = -10
                        if [pos[0] + 2, pos[1]] in self.already_moved:
                            self.intersection_collision_counter += 1
                            self.intersection_crash[i] = True
                    else:
                        pos[0] += 2
                        self.turn[i] = random.randint(0, 1)
                        self.already_moved[i] = pos
                else:
                    if [pos[0] + 1, pos[1] + 1] in self.pos.values():
                        rewards[i] = -5
                        if [pos[0] + 1, pos[1] + 1] in self.already_moved:
                            self.intersection_collision_counter += 1
                            self.intersection_crash[i] = True
                    else:
                        pos[0] += 1
                        pos[1] += 1
                        self.turn[i] = random.randint(0, 1)
                        self.already_moved[i] = pos
            else:
                # vertical lane
                if pos[0] % self.road_size == self.road_size - 1:
                    if [pos[0], pos[1] + 1] in self.pos.values():
                        rewards[i] = -1
                    else:
                        pos[1] += 1
                        self.already_moved[i] = pos
                # horizontal lane
                else:
                    if [pos[0] + 1, pos[1]] in self.pos.values():
                        rewards[i] = -1
                    else:
                        pos[0] += 1
                        self.already_moved[i] = pos

        # print(self.pos)
        # input("cnt?")
        self.prev_rewards = rewards
        states = self.get_state()
        return states, rewards, terminal, {"collisions": self.intersection_collision_counter,
                                           "stalls": self.stall_counter,
                                           "state_central": self.get_central_state(states)}

    def get_central_state(self, states):
        central_state = []
        for i, state in enumerate(states):
            central_state.append([self.turn[i] * 2 - 1,
                                  state[1], self.turn[state[1]] * 2 - 1 if state[1] != -1 else 0,
                                  state[2], self.turn[state[2]] * 2 - 1 if state[1] != -1 else 0,
                                  state[3], self.turn[state[3]] * 2 - 1 if state[1] != -1 else 0])
        return central_state

    # computes each car's observations
    def get_state(self):
        obs = []

        for i, pos in enumerate(self.pos.values()):
            before = -1
            after = -1
            turn = -1

            if pos[0] < 0 or pos[0] >= self.map_border or pos[1] < 0 or pos[1] >= self.map_border:
                obs.append([0, -1, -1, -1])
                continue

            # vertical lane
            if pos[0] % self.road_size == self.road_size - 1:

                # before crossroads
                if pos[1] % self.road_size == self.road_size - 2:
                    before_pos = [pos[0], pos[1] - 1]
                    turn_pos = [pos[0] - 1, pos[1] + 1]
                    if self.turn[i] == 0:
                        after_pos = [pos[0], pos[1] + 2]
                    else:
                        after_pos = [pos[0] + 1, pos[1] + 1]

                # after crossroads
                elif pos[1] % self.road_size == 0:
                    before_pos = [pos[0], pos[1] - 2]
                    after_pos = [pos[0], pos[1] + 1]
                    turn_pos = None

                # lane
                else:
                    # at vertical lane
                    before_pos = [pos[0], pos[1] - 1]
                    after_pos = [pos[0], pos[1] + 1]
                    turn_pos = None

            # horizontal lane
            else:
                # before crossroads
                if pos[0] % self.road_size == self.road_size - 2:
                    before_pos = [pos[0] - 1, pos[1]]
                    turn_pos = [pos[0] + 1, pos[1] - 1]
                    if self.turn[i] == 0:
                        after_pos = [pos[0] + 2, pos[1]]
                    else:
                        after_pos = [pos[0] + 1, pos[1] + 1]

                # after crossroads
                elif pos[0] % self.road_size == 0:
                    before_pos = [pos[0] - 2, pos[1]]
                    after_pos = [pos[0] + 1, pos[1]]
                    turn_pos = None

                # lane
                else:
                    # at vertical lane
                    before_pos = [pos[0] - 1, pos[1]]
                    after_pos = [pos[0] + 1, pos[1]]
                    turn_pos = None

            for j, pos in enumerate(self.pos.values()):
                if after_pos == pos:
                    after = j
                elif before_pos == pos:
                    before = j
                elif turn_pos == pos:
                    turn = j

            # turning, before, after, left
            # obs.append([self.turn[i], before, after, turn])
            obs.append([self.turn[i] * 2 - 1, before, after, turn])

        return obs

    def close(self):
        return

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]
